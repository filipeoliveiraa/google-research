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

"""Shared utilities for behavioral dispositions evaluation."""

import re
import jinja2


def sanitize_model_name(model_name):
  """Sanitizes a model name to be used as a directory or file name."""
  return re.sub(r'[^a-zA-Z0-9]', '_', model_name).lower()


def render_prompt(prompt_template, **kwargs):
  """Renders a Jinja2 template with strict undefined checking.

  Args:
    prompt_template: A Jinja2 template string.
    **kwargs: Template variables.

  Returns:
    The rendered prompt string.
  """

  template = jinja2.Template(prompt_template, undefined=jinja2.StrictUndefined)
  return template.render(**kwargs)


def strip_thinking_tokens(response):
  """Strips thinking tokens (e.g. from Qwen models) from a response."""
  think_end_tag = '</think>'
  think_end = response.find(think_end_tag)
  if think_end != -1:
    response = response[think_end + len(think_end_tag) :]
  return response.strip()
