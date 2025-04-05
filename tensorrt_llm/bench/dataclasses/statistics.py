from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, computed_field


class RequestRecord(BaseModel):
    id: int = -1
    num_input_tokens: int = -1
    tokens: List[int] = []
    error_tokens: int = 0
    start_timestamp: int = -1
    first_token_timestamp: int = -1
    end_timestamp: int = -1
    decode_iteration: int = 0

    def register_event(self,
                       is_error: bool,
                       is_final: bool,
                       timestamp: int,
                       decoding_iter: int,
                       tokens: List[int],
                       first_token_timestamp: int = None) -> None:
        if is_final:
            self.end_timestamp = timestamp
        elif self.first_token_timestamp == -1:
            self.first_token_timestamp = timestamp

        if first_token_timestamp is not None and is_final:
            self.first_token_timestamp = first_token_timestamp

        if is_error:
            self.error_tokens += 1

        self.tokens += tokens
        self.decode_iteration = decoding_iter

    @computed_field
    def num_total_output_tokens(self) -> int:
        """
        Returns the total number of output tokens generated by the request.
        """
        return len(self.tokens)

    @computed_field
    def num_generated_tokens(self) -> int:
        """
        Returns the number of generated (OSL - 1) tokens by the request.
        """
        return self.num_total_output_tokens - 1

    @computed_field
    def generation_time(self) -> int:
        """
        Returns the generation time of the request (E2E Latency - TTFT).
        """
        return self.end_to_end_latency - self.time_to_first_token

    @computed_field
    def time_to_first_token(self) -> int:
        return (self.first_token_timestamp -
                self.start_timestamp if self.first_token_timestamp > 0 else 0.0)

    @computed_field
    def intertoken_latency(self) -> float:
        """
        Returns the time-per-output-token latency of the request [(OSL - 1) / (E2E Latency - TTFT)].
        """
        return ((self.end_timestamp - self.first_token_timestamp) /
                self.num_generated_tokens
                if self.num_generated_tokens > 0 else 0.0)

    @computed_field
    def end_to_end_latency(self) -> int:
        """
        Returns the end-to-end latency of the request (end time - start time).
        """
        return self.end_timestamp - self.start_timestamp

    @computed_field
    def output_token_throughput(self) -> float:
        """
        Returns the total token throughput of the request (Total output tokens / E2E Latency).
        """
        return float(self.num_total_output_tokens) / self.end_to_end_latency

    @computed_field
    def generation_token_throughput(self) -> float:
        return (self.num_generated_tokens / self.generation_time)


class PercentileStats(BaseModel):
    p50: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    average: float

    @classmethod
    def from_iterable(cls, values: List[Any]) -> PercentileStats:
        length = len(values)
        sorted_values = sorted(values)
        return cls(
            p50=sorted_values[int(length * 0.50)],
            p90=sorted_values[int(length * 0.90)],
            p95=sorted_values[int(length * 0.95)],
            p99=sorted_values[int(length * 0.99)],
            average=float(sum(values)) / length,
            minimum=min(values),
            maximum=max(values),
        )


class BenchmarkStatistics(BaseModel):
    # Time-related Properties
    total_latency_ns: float

    # Token-related Properties
    total_output_tokens: int
    total_input_tokens: int

    # General Information
    num_requests: int
    issue_rate_ns: float

    # Speculative Information
    acceptance_rate: float

    # Percentile-related Statistics
    request_latency_percentiles: Optional[PercentileStats] = None
    output_throughput_percentiles: Optional[PercentileStats] = None
    token_percentiles: Optional[PercentileStats] = None
    tpot_percentiles: Optional[PercentileStats] = None
    ttft_percentiles: Optional[PercentileStats] = None
    generation_tp_percentiles: Optional[PercentileStats] = None
    generation_latency_percentiles: Optional[PercentileStats] = None
    acceptance_percentiles: Optional[PercentileStats] = None

    @computed_field
    def sum_per_request_latencies_ns(self) -> float:
        return self.request_latency_percentiles.average * self.num_requests

    @computed_field
    def avg_concurrent_requests(self) -> int:
        return self.sum_per_request_latencies_ns / self.total_latency_ns

    @computed_field
    def generation_tokens(self) -> int:
        return int(self.total_output_tokens - self.num_requests)

    @computed_field
    def total_generation_time_ns(self) -> float:
        return self.generation_latency_percentiles.average * self.num_requests

    @computed_field
    def per_user_time_per_output_token_ns(self) -> float:
        return self.tpot_percentiles.average

    @computed_field
    def per_user_time_to_first_token_ns(self) -> float:
        return self.ttft_percentiles.average

    @computed_field
    def per_user_generation_token_throughput_ns(self) -> float:
        return 1.0 / self.tpot_percentiles.average

    @computed_field
    def request_throughput_ns(self) -> float:
        return float(self.num_requests) / self.total_latency_ns

    @computed_field
    def average_input_length(self) -> float:
        return float(self.total_input_tokens) / self.num_requests

    @computed_field
    def average_output_length(self) -> float:
        return float(self.total_output_tokens) / self.num_requests

    @computed_field
    def output_throughput_tok_ns(self) -> float:
        return float(self.total_output_tokens) / self.total_latency_ns

    @computed_field
    def total_token_throughput_tok_ns(self) -> float:
        return float(self.total_input_tokens +
                     self.total_output_tokens) / self.total_latency_ns

    @computed_field
    def output_throughput_tok_ns_per_user(self) -> float:
        return self.output_throughput_percentiles.average
