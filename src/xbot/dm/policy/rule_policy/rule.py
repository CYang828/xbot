import random

from xbot.util.policy_util import Policy
from xbot.util.db_query import Database


class RulePolicy(Policy):
    def __init__(self):
        super(RulePolicy, self).__init__()
        self.db = Database()

    def predict(self, state):
        belief_state = state["belief_state"]
        cur_domain = state["cur_domain"]
        usr_das = state["user_action"]
        db_res = self.db.query(belief_state, cur_domain)

        sys_da = []
        request_slots = state["request_slots"]
        for domain, slot in request_slots:
            if domain != cur_domain:
                continue
            if not db_res:
                return [["NoOffer", domain, "none", "none"]]
            item_attr = db_res[0][1]
            value = ""
            if slot not in item_attr:
                if "酒店设施" in slot:
                    facility = slot.split("-")[1]
                    value = "是" if facility in item_attr["酒店设施"] else "否"
                    sys_da.append(["Inform", domain, slot, value])
                    continue
            else:
                value = item_attr[slot]
            if slot == "名称":
                if len(db_res) >= 3:
                    for res in db_res[:5]:
                        sys_da.append(["Recommend", domain, slot, res[1][slot]])
                else:
                    sys_da.append(["Inform", domain, slot, item_attr[slot]])
                    # flag = True
                    # for intent, domain, slot, value in usr_das:
                    #     if intent == 'Inform' and slot != '推荐菜' and not slot.startswith('周边'):
                    #         # 如果搜索出的结果并不完全匹配用户的要求，则需要告知要求不同的属性
                    #         if '-' not in slot:
                    #             if item_attr[slot] != value:
                    #                 if '-' in value:
                    #                     pass
                    #                 if flag:  # 表示用户原始要求无法满足
                    #                     sys_da.append(['NoOffer', domain, 'none', 'none'])
                    #                     flag = False
                    #                 if item_attr[slot] is not None:  # 告知新属性值
                    #                     v = str(item_attr[slot])
                    #                     if slot == '门票':
                    #                         v += '元'
                    #                     sys_da.append(['Inform', domain, slot, v])
                    #         else:
                    #             slot, facility = slot.split('-')
                    #             facilities = item_attr[slot]
                    #             if facility not in facilities:
                    #                 if flag:  # 表示用户原始要求无法满足
                    #                     sys_da.append(['NoOffer', domain, 'none', 'none'])
                    #                     flag = False
                    # TODO 告知哪个 facility
            elif slot == "推荐菜" or slot.startswith("周边"):
                for v in value:
                    sys_da.append(["Inform", domain, slot, v])
            # elif slot.startswith('周边'):
            #     for v in value[:int(len(value) * 0.7)]:
            #         sys_da.append(['Inform', domain, slot, v])
            elif slot == "评分":
                if value is None:
                    value = "无"
                else:
                    value = str(int(value)) if value == int(value) else str(value)
                sys_da.append(["Inform", domain, slot, value + "分"])
            elif slot == ["价格", "人均消费"]:
                sys_da.append(["Inform", domain, slot, str(value) + "元"])
            elif slot == "门票":
                if value is None or int(value) == 0:
                    value = "免费"
                elif int(value) > 0:
                    value = str(value) + "元"
                sys_da.append(["Inform", domain, slot, value])
            elif slot == "电话":
                value = value.split(",")
                value = " ".join(value)
                sys_da.append(["Inform", domain, slot, value])
            else:
                sys_da.append(["Inform", domain, slot, value])

        for intent, domain, slot, value in usr_das:
            # 提高 f1 和 recall, joint acc 降低
            # if intent == 'Inform' and slot == '名称':
            #     sys_da.append([intent, domain, slot, value])
            if intent == "General":
                # if domain == 'greet':
                #     sys_da.append(['General', 'greet', 'none', 'none'])
                if domain == "bye" and state["terminated"]:
                    sys_da.append(["General", "bye", "none", "none"])
                if domain == "thank" and state["terminated"]:
                    sys_da.append(["General", "welcome", "none", "none"])
        return sys_da
