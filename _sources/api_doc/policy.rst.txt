policy
###########

.. toctree::
    :maxdepth: 2


BaseCarlaPolicy
================
.. autoclass:: core.policy.base_carla_policy.BaseCarlaPolicy
    :members:


AutoPIDPolicy
================
.. autoclass:: core.policy.AutoPIDPolicy
    :members: _forward_collect, _forward_eval, _reset_collect, _reset_eval


AutoMPCPolicy
================
.. autoclass:: core.policy.AutoMPCPolicy
    :members: _forward_collect, _forward_eval, _reset_collect, _reset_eval


CILRSPolicy
=============
.. autoclass:: core.policy.CILRSPolicy
    :members:


LBCBirdviewPolicy
====================
.. autoclass:: core.policy.LBCBirdviewPolicy
    :members: _forward_eval, _reset_eval


LBCImagePolicy
====================
.. autoclass:: core.policy.LBCImagePolicy
    :members: _forward_eval, _reset_eval