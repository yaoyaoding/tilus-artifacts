from typing import Optional
import importlib.util
import importlib.metadata
import packaging.version
import packaging.specifiers


def check_package_installed(pkg_name: str, version_requirement: Optional[str] = None) -> bool:
    spec = importlib.util.find_spec(pkg_name)
    if spec is None:
        return False
    if version_requirement is not None:
        installed_version = importlib.metadata.version(pkg_name)
        if not packaging.version.parse(installed_version) in packaging.specifiers.SpecifierSet(version_requirement):
            return False
    return True
