from pathlib import Path

def version(base_path='./experiments/my_experiment'):
    base_path = Path(base_path)
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
    version_prefix = 'version_'
    versions = [int(p.name[-1]) for p in base_path.iterdir() if p.is_dir() and p.name.startswith(version_prefix)]
    next_version = max(versions) + 1 if versions else 0
    return version_prefix + str(next_version)