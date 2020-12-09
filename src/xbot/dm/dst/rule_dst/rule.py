from copy import deepcopy
from collections import Counter

from pprint import pprint

from xbot.util.dst_util import DST
from xbot.util.state import default_state


class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states."""

    def __init__(self):
        super(RuleDST, self).__init__()
        self.state = default_state()

    def init_session(self, state=None):
        """Initialize ``self.state`` with a default state.
        :state: see xbot.util.state.default_state
        """
        self.state = default_state() if not state else deepcopy(state)

    def update(self, usr_da=None):
        """update belief_state, cur_domain, request_slot
        :param usr_da: List[List[intent, domain, slot, value]]
        :return: state
        """
        sys_da = self.state["system_action"]

        # 统计各个意图下的 domain
        select_domains = Counter([x[1] for x in usr_da if x[0] == "Select"])
        request_domains = Counter([x[1] for x in usr_da if x[0] == "Request"])
        inform_domains = Counter([x[1] for x in usr_da if x[0] == "Inform"])
        sys_domains = Counter([x[1] for x in sys_da if x[0] in ["Inform", "Recommend"]])

        # 确定domain
        # 为什么首选 select_domain
        # 观察数据集可以发现，Select 意图出现的时候是主导整句话的，即便出现了
        # Inform 和 Request 也是为了为 Select 提供辅助信息，Inform 排在 Request 之后，也是这个道理
        if len(select_domains) > 0:
            self.state["cur_domain"] = select_domains.most_common(1)[0][0]
        elif len(request_domains) > 0:
            self.state["cur_domain"] = request_domains.most_common(1)[0][0]
        elif len(inform_domains) > 0:
            self.state["cur_domain"] = inform_domains.most_common(1)[0][0]
        elif len(sys_domains) > 0:
            self.state["cur_domain"] = sys_domains.most_common(1)[0][0]
        else:
            self.state["cur_domain"] = None

        # 当 system action 中没有 inform 的意图且存在 NoOffer 的意图，判定为确实没 offer
        # 要满足没有 Inform，是因为如果存在 Inform，意味着 system 基于之前的约束信息可能提出了新的考虑意见，
        # 所以只有当 system 基于当前约束完全无法给出建议的时候，当前 domain 的约束才算失效
        no_offer = "NoOffer" in [x[0] for x in sys_da] and "Inform" not in [
            x[0] for x in sys_da
        ]
        # DONE: clean cur domain constraints because no offer
        if no_offer:
            # ISSUE: 过于暴力的做法
            if self.state[
                "cur_domain"
            ]:  # 没有 offer 则清空对应 domain 的 state，即 slot 对应的 value
                self.state["belief_state"][self.state["cur_domain"]] = deepcopy(
                    default_state()["belief_state"][self.state["cur_domain"]]
                )

        # DONE: clean request slot
        # 上一个 system action 中的 inform 和 recommend 表示已经完成的 request，所以从 request_slots 移除
        for domain, slot in deepcopy(self.state["request_slots"]):
            # 已解决的slot全部清除过于暴力
            if [domain, slot] in [
                x[1:3] for x in sys_da if x[0] in ["Inform", "Recommend"]
            ]:
                self.state["request_slots"].remove([domain, slot])

        # DONE: domain switch
        # 非常强依赖前面的NLU
        for intent, domain, slot, value in usr_da:
            # ["Select", "酒店", "源领域", "餐馆"] 找餐馆附近的酒店，所以餐馆是 from_domain
            if intent == "Select":
                from_domain = value
                name = self.state["belief_state"][from_domain]["名称"]
                if name:
                    # 如果当前 domain 等于前一个 domain 那么信息清零再更新，
                    # 与源领域一致说明同样的 domain 要换个具体值了，所以前面的信息没用了
                    if domain == from_domain:
                        self.state["belief_state"][domain] = deepcopy(
                            default_state()["belief_state"][domain]
                        )
                    self.state["belief_state"][domain][
                        "周边{}".format(from_domain)
                    ] = name

        for intent, domain, slot, value in usr_da:
            if intent == "Inform":
                if slot in [
                    "名称",
                    "游玩时间",
                    "酒店类型",
                    "出发地",
                    "目的地",
                    "评分",
                    "门票",
                    "价格",
                    "人均消费",
                ]:
                    self.state["belief_state"][domain][slot] = value
                elif slot == "推荐菜":
                    # 推荐菜可以有多个
                    if not self.state["belief_state"][domain][slot]:
                        self.state["belief_state"][domain][slot] = value
                    else:
                        self.state["belief_state"][domain][slot] += " " + value
                elif "酒店设施" in slot:  # ["Inform", "酒店", "酒店设施-吹风机", "是"]
                    if value == "是":
                        facility = slot.split("-")[1]
                        if not self.state["belief_state"][domain]["酒店设施"]:
                            self.state["belief_state"][domain]["酒店设施"] = facility
                        else:
                            self.state["belief_state"][domain]["酒店设施"] += " " + facility
            elif intent == "Request":  # 存入新增的 domain-slot
                self.state["request_slots"].append([domain, slot])

        # ISSUE： 词槽澄清
        # ISSUE： Repeat Intent
        # ISSUE： System State
        # ISSUE： Hidden Slot
        # ISSUE： 平级槽和依赖槽

        return self.state


if __name__ == "__main__":
    dst = RuleDST()
    dst.init_session()
    # ISSUE： Memory问题
    # input: intent + slot

    dst.update([["Inform", "酒店", "评分", "4分以上"], ["Request", "酒店", "地址", ""]])
    dst.update([["Inform", "酒店", "名称", "颐和园大酒店"], ["Request", "酒店", "地址", "颐和园"]])

    # output: state dict
    pprint(dst.state)
