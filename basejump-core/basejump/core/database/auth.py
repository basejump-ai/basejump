"""Endpoints for user and clients"""

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums, errors

logger = set_logging(handler_option="stream", name=__name__)


def check_user_permissions(required_role: enums.UserRoles, user_role: enums.UserRoles):
    """Verify the user has sufficient permissions"""
    required_permission_lvl = enums.USER_ROLES_LVL_LKUP[required_role.value]
    user_permission_lvl = enums.USER_ROLES_LVL_LKUP[user_role.value]
    try:
        assert user_permission_lvl >= required_permission_lvl
    except AssertionError:
        logger.error("Reqd permission lvl %s", required_permission_lvl)
        logger.error("User permission lvl %s", user_permission_lvl)
        raise errors.UnauthorizedUserRole


def check_result_edit_permission(required_role: enums.UserRoles, user_role: enums.UserRoles) -> None:
    try:
        check_user_permissions(required_role=required_role, user_role=user_role)
    except errors.UnauthorizedUserRole:
        unauth_user = errors.UnauthorizedUserVerifyRole(role_level=required_role)
        raise unauth_user


def check_can_verify(required_role: enums.UserRoles, user_role: enums.UserRoles) -> bool:
    try:
        check_result_edit_permission(required_role=required_role, user_role=user_role)
        can_verify = True
    except errors.UnauthorizedUserVerifyRole:
        can_verify = False
    return can_verify
