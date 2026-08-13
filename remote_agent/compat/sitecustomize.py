# Auto-installed by the remote_agent recipe via PYTHONPATH (RayCluster env).
#
# Python's site module imports any module named `sitecustomize` found on
# sys.path at interpreter startup, so dropping this file into a PYTHONPATH
# directory of the Ray pods makes it run in *every* python process of the job.
# That is needed because verl overrides the per-actor runtime_env when creating
# WorkerDicts (verl/single_controller/ray/base.py), which drops the job-level
# `worker_process_setup_hooks` that main.py registers.
#
# Its only job is to install the config-compat shim (see remote_agent/compat)
# when verl.utils.config is first imported, so core verl config conversion
# tolerates the recipe-added actor_rollout_ref.rollout.remote_agent section.
# Agent-loop registration is *not* handled here: it goes through verl's own
# `rollout.agent.agent_loop_config_path` (see config/agent_loop.yaml).
#
# The hook is lazy (a sys.meta_path finder) and stays side-effect free for
# processes that never import verl.
try:  # pragma: no cover - environment bootstrap
    import sys as _sys

    if "verl.utils.config" not in _sys.modules:

        class _RemoteAgentCompatLoaderWrap:
            """Wraps the real loader to install the shim right after exec."""

            def __init__(self, loader):
                self._loader = loader

            def __getattr__(self, name):
                return getattr(self._loader, name)

            def create_module(self, spec):
                if hasattr(self._loader, "create_module"):
                    return self._loader.create_module(spec)
                return None

            def exec_module(self, module):
                self._loader.exec_module(module)
                try:
                    for _p in ("/workspace",):
                        if _p not in _sys.path:
                            _sys.path.insert(0, _p)
                    from remote_agent.compat import install as _install

                    _install()
                except Exception:
                    pass

        class _RemoteAgentCompatMeta:
            """meta_path finder that only observes verl.utils.config imports."""

            def find_spec(self, fullname, path=None, target=None):
                if fullname != "verl.utils.config":
                    return None
                for finder in _sys.meta_path:
                    if finder is self:
                        continue
                    find = getattr(finder, "find_spec", None)
                    if find is None:
                        continue
                    spec = find(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        spec.loader = _RemoteAgentCompatLoaderWrap(spec.loader)
                        return spec
                return None

        _sys.meta_path.insert(0, _RemoteAgentCompatMeta())
except Exception:
    pass
