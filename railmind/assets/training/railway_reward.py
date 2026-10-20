from swift.rewards import ORM, orms
from railmind.reward import category_reward


class RailwayCategoryReward(ORM):
    def __call__(self, completions, solution, **kwargs):
        if len(completions) != len(solution):
            raise ValueError("Reward completion/solution batch lengths differ")
        return [category_reward(c, s) for c, s in zip(completions, solution)]


orms["railway_category"] = RailwayCategoryReward
