# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Gemini AI Studio API runner for behavioral dispositions evaluation.

This module provides a simple runner that calls Gemini models via the public
Google AI Studio API (google-genai SDK). It requires an API key which can be
obtained from https://aistudio.google.com/apikey.

Usage:
  from google import genai
  client = genai.Client(api_key="YOUR_API_KEY")
  runner = GeminiRunner(client, model_name="gemini-2.0-flash")
  response = runner.generate("Hello, world!")
"""

import dataclasses
import time

from google import genai
from google.genai import types

from . import utils

# Default retry configuration.
_MAX_RETRIES = 5
_INITIAL_DELAY_SEC = 2.0
_MAX_DELAY_SEC = 60.0


@dataclasses.dataclass(frozen=True)
class GenerationOutput:
  """Holds the output of a single generation call."""

  prompt: str
  response: str | None = None  # None if an exception was raised.
  exception: str = ''


class GeminiRunner:
  """A runner that calls Gemini models via the Google AI Studio API."""

  def __init__(
      self,
      client,
      model_name = 'gemini-2.0-flash',
      timeout_ms = 60_000,
  ):
    """Initializes the GeminiRunner.

    Args:
      client: A google.genai.Client instance configured with an API key.
      model_name: The Gemini model name (e.g., 'gemini-2.0-flash').
      timeout_ms: HTTP timeout per API call in milliseconds.
    """
    self._client = client
    self._model_name = model_name
    self._timeout_ms = timeout_ms

  def generate(
      self,
      prompt,
      max_output_tokens = 2048,
      temperature = 1.0,
  ):
    """Generates a response for a single prompt with retry logic.

    Args:
      prompt: The input prompt string.
      max_output_tokens: Maximum number of tokens in the response.
      temperature: Sampling temperature.

    Returns:
      A GenerationOutput with the prompt and response (or exception).
    """
    delay = _INITIAL_DELAY_SEC
    for attempt in range(_MAX_RETRIES):
      try:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                http_options={'timeout': self._timeout_ms},
            ),
        )
        response_text = response.text or ''
        response_text = utils.strip_thinking_tokens(response_text)
        return GenerationOutput(prompt=prompt, response=response_text)
      except Exception as e:  # pylint: disable=broad-except
        if attempt == _MAX_RETRIES - 1:
          return GenerationOutput(prompt=prompt, exception=str(e))
        time.sleep(delay)
        delay = min(delay * 2, _MAX_DELAY_SEC)

    # Should not reach here, but just in case.
    return GenerationOutput(prompt=prompt, exception='Max retries exceeded.')

  def get_name(self):
    """Returns the name of the model."""
    return self._model_name
