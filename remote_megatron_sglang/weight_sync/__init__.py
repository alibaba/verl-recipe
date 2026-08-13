# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Pluggable weight-sync transports. Importing this package registers all
built-in transports with :class:`WeightSyncRegistry`."""

from recipe.remote_megatron_sglang.weight_sync import mooncake, nccl_http, noop, store  # noqa: F401
from recipe.remote_megatron_sglang.weight_sync.base import WeightSyncRegistry, WeightSyncTransport

__all__ = ["WeightSyncRegistry", "WeightSyncTransport"]
