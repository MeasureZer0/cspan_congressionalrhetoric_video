#!/usr/bin/env python3
# Uninstall packages and their dependencies if no other installed package requires them.
import subprocess
import sys
from typing import Dict, Optional

import pkg_resources
from colorama import Fore, init

init(autoreset=True)

if len(sys.argv) < 2:
    print(Fore.CYAN + f"Usage: {sys.argv[0]} pkg1 [pkg2 ...]")
    exit(1)


def get_all_dependencies(
    pkg_name: str,
    installed: Dict[str, pkg_resources.Distribution],
    seen: Optional[set] = None,
) -> set:
    """
    Recursively get all dependencies for a package.
    """
    if seen is None:
        seen = set()
    if pkg_name not in installed or pkg_name in seen:
        return seen
    seen.add(pkg_name)
    dist = installed[pkg_name]
    for req in dist.requires():
        dep_name = req.key
        get_all_dependencies(dep_name, installed, seen)
    return seen


# Get all installed packages
installed = {dist.key: dist for dist in pkg_resources.working_set}

# Get all dependencies for the target packages
to_remove = set()
for pkg in sys.argv[1:]:
    to_remove |= get_all_dependencies(pkg.lower(), installed)

# Find all packages required by others
required_by_others = set()
for dist in installed.values():
    for req in dist.requires():
        dep_name = req.key
        if dep_name in to_remove and dist.key not in to_remove:
            required_by_others.add(dep_name)

# Only remove packages not required by others
final_remove = [pkg for pkg in to_remove if pkg not in required_by_others]
if not final_remove:
    print(Fore.RED + "No packages can be safely removed.")
    sys.exit(0)
print(Fore.YELLOW + "Uninstalling packages and dependencies:", *final_remove)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", *final_remove], check=True
)
print(Fore.GREEN + "Uninstallation complete.")

print(Fore.YELLOW + "Generating requirements.txt")
subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    stdout=open("requirements.txt", "w"),
    check=True,
)
print(Fore.GREEN + "requirements.txt updated.")
