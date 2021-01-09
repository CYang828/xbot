from copy import deepcopy
from typing import List, Optional

from transitions import Machine

from xbot.util.dst_util import DST
from xbot.util.state import default_state


class FSMDST(DST):
    """Only for restaurant domain."""

    def __init__(self):
        super(FSMDST, self).__init__()
        self.dialogue_state = default_state()
        self.dialogue_state["cur_domain"] = "餐馆"
        self.restaurant_states = list(self.dialogue_state["belief_state"]["餐馆"].keys())
        self.restaurant_states.append("Request")
        self.machine = Machine(
            model=self, states=self.restaurant_states, initial="greet"
        )

        self.dest2trigger = {
            "推荐菜": {
                "trigger": "inform_restaurant_dish",
                "callback": "set_recommend_dish",
            },
            "评分": {"trigger": "inform_restaurant_rating", "callback": "set_rating"},
            "人均消费": {
                "trigger": "inform_restaurant_avg_cost",
                "callback": "set_avg_cost",
            },
            "周边酒店": {
                "trigger": "inform_restaurant_surrounding_hotel",
                "callback": "set_surrounding_hotel",
            },
            "周边景点": {
                "trigger": "inform_restaurant_surrounding_attraction",
                "callback": "set_surrounding_attraction",
            },
            "周边餐馆": {
                "trigger": "inform_restaurant_surrounding_restaurant",
                "callback": "set_surrounding_restaurant",
            },
        }

        for slot, trigger in self.dest2trigger.items():
            self.machine.add_transition(
                trigger=trigger["trigger"],
                source="*",
                dest=slot,
                before=trigger["callback"],
            )

    def init_session(self, state: Optional[dict] = None) -> None:
        """Initialize ``self.state`` with a default state.

        Args:
            state: see xbot.util.state.default_state
        """
        self.dialogue_state = default_state() if not state else deepcopy(state)

    def set_recommend_dish(self, dish: str) -> None:
        """Callback function for recommend dish state,
        update dialogue state (recommend dish).

        Args:
            dish: specified dish from user actions predicted by NLU
        """
        self.dialogue_state["belief_state"]["餐馆"]["推荐菜"] = dish

    def set_rating(self, rating: str) -> None:
        self.dialogue_state["belief_state"]["餐馆"]["评分"] = rating

    def set_avg_cost(self, cost: str) -> None:
        self.dialogue_state["belief_state"]["餐馆"]["人均消费"] = cost

    def set_surrounding_hotel(self, hotel: str) -> None:
        self.dialogue_state["belief_state"]["餐馆"]["周边酒店"] = hotel

    def set_surrounding_attraction(self, attraction: str) -> None:
        self.dialogue_state["belief_state"]["餐馆"]["周边景点"] = attraction

    def set_surrounding_restaurant(self, restaurant: str) -> None:
        self.dialogue_state["belief_state"]["餐馆"]["周边餐馆"] = restaurant

    def set_request(self, slot: str) -> None:
        self.dialogue_state["request_slots"].append(["餐馆", slot])

    def update(self, action: List[List[str]]) -> None:
        """Update dialogue stata according to user actions

        Args:
            action: user actions from NLU predictions
        """
        self.update_slot_value(action)

        self.clear_finished_requests()

    def clear_finished_requests(self) -> None:
        """Clear requests informed by system."""
        sys_da = self.dialogue_state["system_action"]
        for domain, slot in deepcopy(self.dialogue_state["request_slots"]):
            if [domain, slot] in [
                x[1:3] for x in sys_da if x[0] in ["Inform", "Recommend"]
            ]:
                self.dialogue_state["request_slots"].remove([domain, slot])

    def update_slot_value(self, action: List[List[str]]) -> None:
        """Update slot in dialogue state.

        Args:
            action: user actions from NLU predictions
        """
        for act in action:
            if act[0] == "General":
                continue

            slot, value = act[2:]
            if act[0] == "Request":
                self.machine.request(slot)
            else:
                self.trigger(self.dest2trigger[slot]["trigger"], value)


if __name__ == "__main__":
    dst = FSMDST()
