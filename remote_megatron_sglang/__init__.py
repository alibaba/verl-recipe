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
"""RemoteBackend that drives an external Megatron training cluster (deployed as
a Kubeflow PyTorchJob) plus an external SGLang inference cluster (deployed as a
RoleBasedGroup), with verl reduced to a CPU-only orchestrator.

See DESIGN.md and README.md in this directory. Importing this package's
``backend`` module registers the ``megatron_sglang`` backend with
:class:`verl.remote_backend.RemoteBackendRegistry`.
"""
