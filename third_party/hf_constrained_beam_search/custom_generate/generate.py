from typing import TYPE_CHECKING, Optional, Union

import logging
import torch
import torch.nn as nn
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamOutput,
    GenerationMixin,
)

from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import ConstrainedBeamSearchScorer

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer


logger = logging.getLogger(__name__)


def _constrained_beam_search(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    if generation_config.constraints is not None or generation_config.force_words_ids is not None:
        constrained_wrong_parameter_msg = (
            "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. "
            "However, `{flag_name}` is set to `{flag_value}`, which is incompatible with this generation "
            "mode. Set `constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue."
        )
        if generation_config.do_sample is True:
            raise ValueError(
                constrained_wrong_parameter_msg.format(
                    flag_name="do_sample",
                    flag_value=generation_config.do_sample,
                )
            )

    final_constraints = []
    if generation_config.constraints is not None:
        final_constraints = generation_config.constraints

    if generation_config.force_words_ids is not None:

        def typeerror():
            raise ValueError(
                "`force_words_ids` has to either be a `list[list[list[int]]]` or `list[list[int]]` "
                f"of positive integers, but is {generation_config.force_words_ids}."
            )

        if (
            not isinstance(generation_config.force_words_ids, list)
            or len(generation_config.force_words_ids) == 0
        ):
            typeerror()

        for word_ids in generation_config.force_words_ids:
            if isinstance(word_ids[0], list):
                if not isinstance(word_ids, list) or len(word_ids) == 0:
                    typeerror()
                if any(not isinstance(token_ids, list) for token_ids in word_ids):
                    typeerror()
                if any(
                    any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                    for token_ids in word_ids
                ):
                    typeerror()

                constraint = DisjunctiveConstraint(word_ids)
            else:
                if not isinstance(word_ids, list) or len(word_ids) == 0:
                    typeerror()
                if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                    typeerror()

                constraint = PhrasalConstraint(word_ids)
            final_constraints.append(constraint)

    constrained_beam_scorer = ConstrainedBeamSearchScorer(
        constraints=final_constraints,
        batch_size=input_ids.shape[0] // generation_config.num_beams,
        num_beams=generation_config.num_beams,
        device=input_ids.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
        max_length=generation_config.max_length,
    )
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate

    batch_size = len(constrained_beam_scorer._beam_hyps)
    num_beams = constrained_beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape[:2]
    if hasattr(model, "_get_initial_cache_position"):
        model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False
    decoder_prompt_len = input_ids.shape[1]
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        outputs = model(**model_inputs, return_dict=True)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            cur_len += 1
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )
        scores_for_all_vocab = next_token_scores.clone()

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = (next_tokens / vocab_size).long()
        next_tokens = next_tokens % vocab_size

        beam_outputs = constrained_beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            scores_for_all_vocab,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        del outputs

        if model_kwargs.get("past_key_values", None) is not None:
            if hasattr(model, "_reorder_cache"):
                model_kwargs["past_key_values"] = model._reorder_cache(
                    model_kwargs["past_key_values"],
                    beam_idx,
                )
            else:
                model_kwargs["past_key_values"].reorder_cache(beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices)))

        cur_len += 1
        if constrained_beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    sequence_outputs = constrained_beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        if model.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return GenerateBeamDecoderOnlyOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
            logits=raw_logits,
            beam_indices=sequence_outputs["beam_indices"],
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    return sequence_outputs["sequences"]


def generate(model, *args, **kwargs):
    return GenerationMixin.generate(model, *args, custom_generate=_constrained_beam_search, **kwargs)
