from pref.utils import define_reward_type

# total welfare (TW) design
@define_reward_type(["tw", "total_welfare"])
def TW(ucb):
    return ucb