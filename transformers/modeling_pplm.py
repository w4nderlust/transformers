# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: refactor to match interfaces of modeling_ctrl.py
# TODO: use config_pplm.py
# TODO: add code for training a custom discriminator

"""
Example command with bag of words:
python modeling_pplm.py -B data/pplm/bow/space.txt --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl-scale 0.01 --gm-scale 0.95

Example command with discriminator:
python modeling_pplm.py -D sentiment --label_class 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl-scale 0.01 --gm-scale 0.95
"""

import argparse
from operator import add
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2-medium")

discriminator_models_params = {
    "clickbait": {
        "path": "data/pplm/discriminators/clickbait_classifierhead.pt",
        "class_size": 5,
        "embed_size": 1034,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
    },
    "sentiment": {
        "path": "data/pplm/discriminators/sentiment_classifierhead.pt",
        "class_size": 2,
        "embed_size": 1034,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
    },
    "toxicity": {
        "path": "data/pplm/discriminators/toxicity_classifierhead.pt",
        "class_size": 2,
        "embed_size": 1034,
        "class_vocab": {"non_toxic": 0, "toxic": 1},
        "default_class": 0,
    },
}


class ClassificationHead(torch.nn.Module):
    """ Classification Head for the transformer """

    def __init__(self, class_size=5, embed_size=2048):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # hidden_state = hidden_state[:, :-1].contiguous().view(-1, self.n_embd)
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k <= 0:
        return logits

    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)

        if probs:
            return torch.where(
                logits < batch_mins, torch.ones_like(logits) * 0.0, logits
            )

        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def perturb_past(
    past,
    model,
    last,
    vocab_size=50257,
    original_probs=None,
    accumulated_hidden=None,
    true_past=None,
    grad_norms=None,
    stepsize=0.01,
    classifier=None,
    label_class=None,
    bow_indices=None,
    loss_type=0,
    num_iterations=3,
    gm_scale=0.9,
    kl_scale=0.01,
    window_length=0,
    horizon_length=1,
    decay=False,
    gamma=1.5,
):
    # collect one hot vectors for bags of words
    # TODO: maybe this could be optimized by doing it only one time
    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).cuda()
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, vocab_size).cuda()
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)

    # Generate inital perturbed past
    past_perturb_orig = [
        (np.random.uniform(0.0, 0.0, p.shape).astype("float32")) for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, current_length, _ = past[0].shape

    if current_length > window_length and window_length > 0:
        ones_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([current_length - window_length])
            + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2
        ).cuda()

    else:
        window_mask = torch.ones_like(past[0]).cuda()

    loss_per_iter = []
    for i in range(num_iterations):
        past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
        past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]

        pert_past = list(map(add, past, past_perturb))

        _, _, _, current_length, _ = past_perturb[0].shape

        # Compute hidden using perturbed past
        logits, future_past, all_hidden = model(last, past=pert_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # TODO: Check the layer-norm consistency of this with trained discriminator
        # logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :]
        probabs = F.softmax(logits, dim=-1)
        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_good in one_hot_bows_vectors:
                good_logits = torch.mm(probabs, torch.t(one_hot_good))
                loss_word = good_logits
                loss_word = torch.sum(loss_word)
                loss_word = -torch.log(loss_word)
                # loss_word = torch.sum(loss_word) /torch.sum(one_hot_good)
                loss += loss_word
                loss_list.append(loss_word)
            print("words", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            new_true_past = true_past
            for i in range(horizon_length):
                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                _, new_true_past, all_hidden = model(future_probabs, past=new_true_past)
                future_hidden = all_hidden[-1]  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    future_hidden, dim=1
                )

            predicted_sentiment = classifier(
                new_accumulated_hidden / (current_length + 1 + horizon_length)
            )

            label = torch.tensor([label_class], device="cuda", dtype=torch.long)
            discrim_loss = ce_loss(predicted_sentiment, label)
            print("discrim", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            p = F.softmax(original_probs[:, -1, :], dim=-1)
            p = (
                p
                + SMALL_CONST
                * (p <= SMALL_CONST).type(torch.FloatTensor).cuda().detach()
            )
            correction = (
                SMALL_CONST
                * (probabs <= SMALL_CONST).type(torch.FloatTensor).cuda().detach()
            )
            corrected_probabs = probabs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probabs * (corrected_probabs / p).log()).sum()
            )
            # print('kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss  # + discrim_loss

        print((loss - kl_loss).data.cpu().numpy())

        loss_per_iter.append(loss.data.cpu().numpy())
        loss.backward()
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(past_perturb)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(past_perturb)
            ]

        grad = [
            -stepsize
            * (p_.grad * window_mask / grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(past_perturb)
        ]
        past_perturb_orig = list(map(add, grad, past_perturb_orig))

        for p_ in past_perturb:
            p_.grad.data.zero_()

        new_past = []
        for p in past:
            new_past.append(p.detach())

        past = new_past

    past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
    past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]
    pert_past = list(map(add, past, past_perturb))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
    name: Optional[str], label_class: Union[str, int], device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = discriminator_models_params[name]
    classifier = ClassificationHead(
        class_size=params["class_size"], embed_size=params["embed_size"]
    ).to(device)
    # TODO why do we need this?
    classifier.eval()

    if isinstance(label_class, str):
        if label_class in params["class_vocab"]:
            label_id = params["class_vocab"][label_class]
        else:
            label_id = params["default_class"]
            print("label_class {} not in class_vocab".format(label_class))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(label_class, int):
        if label_class in set(params["class_vocab"].values()):
            label_id = label_class
        else:
            label_id = params["default_class"]
            print("label_class {} not in class_vocab".format(label_class))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_paths: List[str]) -> List[int]:
    bow_indices = []
    for bag_of_words_path in bag_of_words_paths:
        with open(bag_of_words_path, "r") as f:
            words = f.read().split("\n")
        # TODO: why space concat?
        bow_indices.append([TOKENIZER.encode(" " + word) for word in words])
    return bow_indices


def full_text_generation(model, args, context=None, sample=True, device="cuda"):
    classifier, class_id = get_classifier(args.discrim, args.label_class, device)

    bow_indices = []
    if args.bag_of_words:
        bow_indices = get_bag_of_words_indices(args.bag_of_words.split(";"))

    if args.bag_of_words and classifier:
        print("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif args.bag_of_words:
        loss_type = PPLM_BOW
        print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either --bag_of_words (-B) or --discrim (-D)")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model, context=context, device=device, length=args.length, perturb=False
    )
    torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(args.num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            context=context,
            device=device,
            sample=sample,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            label_class=class_id,
            loss_type=loss_type,
            length=args.length,
            grad_length=args.grad_length,
            stepsize=args.stepsize,
            num_iterations=args.num_iterations,
            temperature=args.temperature,
            gm_scale=args.gm_scale,
            kl_scale=args.kl_scale,
            top_k=args.top_k,
            window_length=args.window_length,
            horizon_length=args.horizon_length,
            decay=args.decay,
            gamma=args.gamma,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
    model,
    context=None,
    past=None,
    device="cuda",
    sample=True,
    perturb=True,
    classifier=None,
    label_class=None,
    bow_indices=None,
    loss_type=0,
    length=100,
    grad_length=10000,
    stepsize=0.02,
    num_iterations=3,
    temperature=1.0,
    gm_scale=0.9,
    kl_scale=0.01,
    top_k=10,
    window_length=0,
    horizon_length=1,
    decay=False,
    gamma=1.5,
):
    output_so_far = (
        torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
        if context
        else None
    )

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    for i in trange(length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token
        # TODO: what's i/p?
        # Therefore, use everything from before current i/p token to generate relevant past

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            _, past, _ = model(output_so_far[:, :-1])
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]

        else:
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            pert_past, _, grad_norms, loss_this_iter = perturb_past(
                past,
                model,
                last,
                original_probs=unpert_logits,
                accumulated_hidden=accumulated_hidden,
                true_past=unpert_past,
                grad_norms=grad_norms,
                stepsize=current_stepsize,
                classifier=classifier,
                label_class=label_class,
                bow_indices=bow_indices,
                loss_type=loss_type,
                num_iterations=num_iterations,
                gm_scale=gm_scale,
                kl_scale=kl_scale,
                window_length=window_length,
                horizon_length=horizon_length,
                decay=decay,
                gamma=gamma,
            )
            loss_in_time.append(loss_this_iter)

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        # compute the discriminator loss using unperturbed hidden
        if classifier is not None:
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([label_class], device="cuda", dtype=torch.long)
            unpert_discrim_loss = torch.nn.CrossEntropyLoss()(prediction, label)
            print("unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model probabilities
        if perturb:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            gm_scale = gm_scale
            pert_probs = (pert_probs ** gm_scale) * (
                unpert_probs ** (1 - gm_scale)
            )  # + SMALL_CONST

            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        )
        print(TOKENIZER.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity"),
        help="Discriminator to use for loss-type 2",
    )
    parser.add_argument(
        "--label_class",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gm-scale", type=float, default=0.9)
    parser.add_argument("--kl-scale", type=float, default=0.01)
    parser.add_argument("--nocuda", action="store_true", help="no cuda")
    parser.add_argument(
        "--uncond", action="store_true", help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake", help="Prefix texts to condition on"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad-length", type=int, default=10000)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
        "0 corresponds to infinite window length",
    )
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)

    args = parser.parse_args()

    # set Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set the device
    device = "cpu" if args.nocuda else "cuda"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(args.model_path, output_hidden_states=True)
    model.to(device)
    # TODO: why is this needed?
    model.eval()

    # freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if args.uncond:
        # TODO: Why two tokens?
        tokenized_cond_text = TOKENIZER.encode(
            [TOKENIZER.eos_token, TOKENIZER.eos_token]
        )
    else:
        raw_text = args.cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = TOKENIZER.encode(TOKENIZER.eos_token + raw_text)

    print("= Prefix of sentence =")
    print(TOKENIZER.decode(tokenized_cond_text))
    print()

    # generate unperturbed and perturbed texts

    # latent_perturb returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model, args=args, context=tokenized_cond_text, device=device
    )

    # untokenize unperturbed text
    unpert_gen_text = TOKENIZER.decode(unpert_gen_tok_text.tolist()[0])

    print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    generated_texts = []

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            unpert_gen_text = TOKENIZER.decode(pert_gen_tok_text.tolist()[0])

            print("= Perturbed generated text {} =".format(i + 1))
            print(unpert_gen_text)
            print()
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        # TODO: why do we need to keep them? Tey are not used anywhere
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return


if __name__ == "__main__":
    run_model()
