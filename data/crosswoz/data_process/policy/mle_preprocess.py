import json

import numpy as np

from src.xbot.util.db_query import Database
from src.xbot.util.state import default_state


def delexicalize_da(da):
    delexicalized_da = []
    counter = {}
    for intent, domain, slot, value in da:
        if intent in ["Inform", "Recommend"]:
            key = "+".join([intent, domain, slot])
            counter.setdefault(key, 0)
            counter[key] += 1
            delexicalized_da.append(key + "+" + str(counter[key]))
        else:
            delexicalized_da.append("+".join([intent, domain, slot, value]))

    return delexicalized_da


def lexicalize_da(da, cur_domain, entities):
    """将泛化的 tag 替换为具体的值"""
    not_dish = {"当地口味", "老字号", "其他", "美食林风味", "特色小吃", "美食林臻选", "深夜营业", "名人光顾", "四合院"}
    lexicalized_da = []
    for a in da:
        intent, domain, slot, value = a.split("+")
        if intent in ["General", "NoOffer"]:
            lexicalized_da.append([intent, domain, slot, value])
        elif domain == cur_domain:
            value = int(value) - 1
            if domain == "出租":
                assert intent == "Inform"
                assert slot in ["车型", "车牌"]
                assert value == 0
                value = entities[0][1][slot]  # 将"车型"的替换为具体的型号
                lexicalized_da.append([intent, domain, slot, value])
            elif domain == "地铁":
                assert intent == "Inform"
                assert slot in ["出发地附近地铁站", "目的地附近地铁站"]
                assert value == 0
                if slot == "出发地附近地铁站":
                    candidates = [v for n, v in entities if "起点" in n]
                    if candidates:
                        value = candidates[0]["地铁"]
                    else:
                        value = "无"
                else:
                    candidates = [v for n, v in entities if "终点" in n]
                    if candidates:
                        value = candidates[0]["地铁"]
                    else:
                        value = "无"
                lexicalized_da.append([intent, domain, slot, value])
            else:
                if intent == "Recommend":
                    assert slot == "名称"
                    if len(entities) > value:
                        value = entities[value][0]
                        lexicalized_da.append([intent, domain, slot, value])
                else:
                    assert intent == "Inform"
                    if len(entities) > value:
                        entity = entities[0][1]
                        if "周边" in slot:
                            assert isinstance(entity[slot], list)
                            if value < len(entity[slot]):
                                value = entity[slot][value]
                                lexicalized_da.append([intent, domain, slot, value])
                        elif slot == "推荐菜":
                            assert isinstance(entity[slot], list)
                            dishes = [x for x in entity[slot] if x not in not_dish]
                            if len(dishes) > value:
                                value = dishes[value]
                            lexicalized_da.append([intent, domain, slot, value])
                        elif "酒店设施" in slot:
                            assert value == 0
                            slot, value = slot.split("-")
                            assert isinstance(entity[slot], list)
                            if value in entity[slot]:
                                lexicalized_da.append(
                                    [intent, domain, "-".join([slot, value]), "是"]
                                )
                            else:
                                lexicalized_da.append(
                                    [intent, domain, "-".join([slot, value]), "否"]
                                )
                        elif slot in ["门票", "价格", "人均消费"]:
                            assert value == 0
                            value = entity[slot]
                            lexicalized_da.append([intent, domain, slot, f"{value}元"])
                        elif slot == "评分":
                            assert value == 0
                            value = entity[slot]
                            lexicalized_da.append([intent, domain, slot, f"{value}分"])
                        else:
                            assert value == 0
                            value = entity[slot]
                            lexicalized_da.append([intent, domain, slot, value])
    return lexicalized_da


class CrossWozVector:
    def __init__(self, sys_da_voc_json, usr_da_voc_json):
        self.sys_da_voc = json.load(open(sys_da_voc_json, encoding="utf8"))
        self.usr_da_voc = json.load(open(usr_da_voc_json, encoding="utf8"))
        self.database = Database()

        self.sys_da2id = {a: i for i, a in enumerate(self.sys_da_voc)}
        self.id2sys_da = {i: a for i, a in enumerate(self.sys_da_voc)}

        # 155
        self.sys_da_dim = len(self.sys_da_voc)

        self.usr_da2id = {a: i for i, a in enumerate(self.usr_da_voc)}
        self.id2usr_da = {i: a for i, a in enumerate(self.usr_da_voc)}

        # 142
        self.usr_da_dim = len(self.usr_da_voc)

        # 26
        self.belief_state_dim = 0  # belief_state 中所有的 slot-values 的数量
        for domain, svs in default_state()["belief_state"].items():
            self.belief_state_dim += len(svs)

        self.db_res_dim = 4

        self.state_dim = (
            self.sys_da_dim
            + self.usr_da_dim
            + self.belief_state_dim
            + self.db_res_dim
            + 1
        )  # terminated

        self.cur_domain = None
        self.belief_state = None
        self.db_res = None

    def state_vectorize(self, state):
        self.belief_state = state["belief_state"]
        self.cur_domain = state["cur_domain"]

        da = state["user_action"]
        da = delexicalize_da(da)
        usr_act_vec = np.zeros(self.usr_da_dim)
        for a in da:
            if a in self.usr_da2id:
                usr_act_vec[self.usr_da2id[a]] = 1.0

        da = state["system_action"]
        da = delexicalize_da(da)
        sys_act_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2id:
                sys_act_vec[self.sys_da2id[a]] = 1.0

        belief_state_vec = np.zeros(self.belief_state_dim)
        i = 0
        for domain, svs in state["belief_state"].items():
            for slot, value in svs.items():
                if value:
                    belief_state_vec[i] = 1.0
                i += 1

        # 以上每一步都将 vec 中出现的 item 置为 1
        # 根据目前的状态，向数据库查询返回符合要求的 item
        self.db_res = self.database.query(state["belief_state"], state["cur_domain"])
        db_res_num = len(self.db_res)
        db_res_vec = np.zeros(4)
        if db_res_num == 0:
            db_res_vec[0] = 1.0
        elif db_res_num == 1:
            db_res_vec[1] = 1.0
        elif 1 < db_res_num < 5:  # 为什么这么分
            db_res_vec[2] = 1.0
        else:
            db_res_vec[3] = 1.0

        terminated = 1.0 if state["terminated"] else 0.0

        # 横向拼接
        state_vec = np.r_[
            usr_act_vec, sys_act_vec, belief_state_vec, db_res_vec, terminated
        ]
        return state_vec

    def action_devectorize(self, action_vec):
        """
        must call state_vectorize func before
        :param action_vec:
        :return:
        """
        da = []
        for i, idx in enumerate(action_vec):
            if idx == 1:
                da.append(self.id2sys_da[i])
        lexicalized_da = lexicalize_da(
            da=da, cur_domain=self.cur_domain, entities=self.db_res
        )
        return lexicalized_da

    def action_vectorize(self, da):
        da = delexicalize_da(da)
        sys_act_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2id:
                sys_act_vec[self.sys_da2id[a]] = 1.0
        return sys_act_vec
