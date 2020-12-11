import os
import re
import json

from xbot.util.state import default_state


def contains(arr, s):
    """
    反向逻辑，当 arr 中没有一个包含 s 的时候才为 true
    """
    return not len(tuple(filter(lambda item: (not (item.find(s) < 0)), arr)))


def contains_human(arr, s):
    return not [a for a in arr if arr.find(s) >= 0]


class Database:
    """docstring for Database"""

    def __init__(self):
        self.data = {}
        db_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data/crosswoz/database",
        )
        with open(os.path.join(db_dir, "metro_db.json"), "r", encoding="utf-8") as f:
            self.data["地铁"] = json.load(f)
        with open(os.path.join(db_dir, "hotel_db.json"), "r", encoding="utf-8") as f:
            self.data["酒店"] = json.load(f)
        with open(
            os.path.join(db_dir, "restaurant_db.json"), "r", encoding="utf-8"
        ) as f:
            self.data["餐馆"] = json.load(f)
        with open(
            os.path.join(db_dir, "attraction_db.json"), "r", encoding="utf-8"
        ) as f:
            self.data["景点"] = json.load(f)

        self.schema = {
            "景点": {
                "名称": {"params": None},
                "门票": {"type": "between", "params": [None, None]},  # 价格范围
                "游玩时间": {"params": None},
                "评分": {"type": "between", "params": [None, None]},  # 评分范围
                "周边景点": {"type": "in", "params": None},
                "周边餐馆": {"type": "in", "params": None},
                "周边酒店": {"type": "in", "params": None},
            },
            "餐馆": {
                "名称": {"params": None},
                "推荐菜": {"type": "multiple_in", "params": None},
                "人均消费": {"type": "between", "params": [None, None]},
                "评分": {"type": "between", "params": [None, None]},
                "周边景点": {"type": "in", "params": None},
                "周边餐馆": {"type": "in", "params": None},
                "周边酒店": {"type": "in", "params": None},
            },
            "酒店": {
                "名称": {"params": None},
                "酒店类型": {"params": None},
                "酒店设施": {"type": "multiple_in", "params": None},
                "价格": {"type": "between", "params": [None, None]},
                "评分": {"type": "between", "params": [None, None]},
                "周边景点": {"type": "in", "params": None},
                "周边餐馆": {"type": "in", "params": None},
                "周边酒店": {"type": "in", "params": None},
            },
            "地铁": {
                "起点": {"params": None},
                "终点": {"params": None},
            },
            "出租": {
                "起点": {"params": None},
                "终点": {"params": None},
            },
        }

    def query(self, belief_state, cur_domain):
        """
        query database using belief state, return list of entities, same format as database
        :param belief_state: state['belief_state']
        :param cur_domain: maintain by DST, current query domain
        :return: list of entities
        """
        if not cur_domain:  # 没有 domain 就是类似于 General+greet+none，无需查询
            return []
        cur_query_form = {}
        # 当前 belief state 的当前 domain
        for slot, value in belief_state[cur_domain].items():
            if not value:  # 没有值也就没有约束
                continue
            if slot == "出发地":
                slot = "起点"  # 出发地在 schema 中是 起点，所以改写
            elif slot == "目的地":
                slot = "终点"
            if slot == "名称":  # 名称确定了，就不用其他条件缩小范围
                # DONE: if name is specified, ignore other constraints
                cur_query_form = {"名称": value}
                break
            elif slot == "评分":
                if re.match(r"(\d\.\d|\d)", value):  # 整数或者小数
                    if re.match(r"\d\.\d", value):
                        score = float(re.match(r"\d\.\d", value)[0])
                    else:
                        score = int(re.match(r"\d", value)[0])
                    cur_query_form[slot] = [score, None]  # 并没有考虑范围结果，为什么使用范围结构
                # else:
                #     assert 0, value
            elif slot in ["门票", "人均消费", "价格"]:
                low, high = None, None
                if re.match(r"(\d+)-(\d+)", value):
                    low = int(re.match(r"(\d+)-(\d+)", value)[1])
                    high = int(re.match(r"(\d+)-(\d+)", value)[2])
                elif re.match(r"\d+", value):
                    if "以上" in value:
                        low = int(re.match(r"\d+", value)[0])
                    elif "以下" in value:
                        high = int(re.match(r"\d+", value)[0])
                    else:
                        low = high = int(re.match(r"\d+", value)[0])
                elif slot == "门票":
                    if value == "免费":
                        low = high = 0
                    elif value == "不免费":
                        low = 1
                    else:
                        print(value)  # 这个 value 为什么直接丢弃了，数据集中是存在门票价格范围的样本的
                        # assert 0
                cur_query_form[slot] = [low, high]
            else:
                cur_query_form[slot] = value
        cur_res = self.query_schema(field=cur_domain, args=cur_query_form)
        if cur_domain == "出租":
            res = [cur_res]
        elif cur_domain == "地铁":
            res = []
            for r in cur_res:
                # 只取了随机一条路线
                if not res and "起点" in r[0]:
                    res.append(r)
                    break
            for r in cur_res:
                if "终点" in r[0]:
                    res.append(r)
                    break
        else:
            res = cur_res

        return res

    def query_schema(self, field, args):
        if field not in self.schema:
            raise Exception(f"Unknown field {field}")
        if not isinstance(args, dict):
            raise Exception("`args` must be dict")
        db = self.data.get(field)
        plan = self.schema[field]
        for key, value in args.items():  # key: slot
            if key not in plan:
                raise Exception(f"Unknown key {key}")
            value_type = plan[key].get("type")
            if value_type == "between":
                if value[0] is not None:
                    plan[key]["params"][0] = float(value[0])
                if value[1] is not None:
                    plan[key]["params"][1] = float(value[1])
            else:
                if not isinstance(value, str):
                    raise Exception("Value for `%s` must be string" % key)
                plan[key]["params"] = value
        if field in ["地铁", "出租"]:
            s = plan["起点"]["params"]
            e = plan["终点"]["params"]
            if not s or not e:  # 起点和终点一个没确定就无法形成约束查处结果
                return []
            if field == "出租":
                return [f"出租 ({s} - {e})", {"领域": "出租", "车型": "#CX", "车牌": "#CP"}]
            else:

                def func1(item):
                    if item[0].find(s) >= 0:
                        return [f"(起点) {item[0]}", item[1]]

                def func2(item):
                    if item[0].find(e) >= 0:
                        return [f"(终点) {item[0]}", item[1]]
                    return None

                starts = [func1(start) for start in db if func1(start) is not None]
                ends = [func2(end) for end in db if func2(end) is not None]

                return starts + ends

        def func3(item):
            details = item[1]
            for slot, _ in args.items():
                db_values = details.get(slot)  # 数据库中的值
                absence = db_values is None
                options = plan[slot]
                if options.get("type") == "between":
                    left = options["params"][0]
                    right = options["params"][1]
                    if left is not None:
                        if absence:  # 约束中包含的， 数据库中的当前条目没有，所以不满足
                            return False
                    else:
                        left = float("-inf")
                    if right is not None:
                        if absence:  # 同 left
                            return False
                    else:
                        right = float("inf")
                    # 等于 between 但是没有实际的上下限约束，只要是个正常的实数就行
                    if left > db_values or db_values > right:
                        return False
                elif options.get("type") == "in":  # 周边xx，val 为列表
                    plan_values = options["params"]
                    if plan_values is not None:
                        if absence:
                            return False
                        if contains(db_values, plan_values):  # contains 当不包含时为 true
                            return False
                elif options.get("type") == "multiple_in":
                    plan_values = options["params"]
                    if plan_values is not None:
                        if absence:
                            return False
                        s_arr = [val for val in plan_values.split(" ") if val]
                        # 只要有一个推荐菜没有被包含就视为不满足约束
                        if [val for val in s_arr if contains(db_values, val)]:
                            return False
                else:
                    plan_values = options["params"]
                    if plan_values is not None:
                        if absence:
                            return False
                        if db_values.find(plan_values) < 0:
                            return False
            return True

        return [item for item in db if func3(item)]


if __name__ == "__main__":
    from pprint import pprint
    from collections import Counter

    database = Database()
    state = default_state()
    dishes = {}
    for n, v in database.query(state["belief_state"], "餐馆"):
        for dish in v["推荐菜"]:
            dishes.setdefault(dish, 0)
            dishes[dish] += 1
    pprint(Counter(dishes))
