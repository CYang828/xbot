import os
import json

from xbot.util.path import get_data_path
from xbot.util.file_util import dump_json
from xbot.dm.dst.rule_dst.rule import RuleDST
from xbot.dm.policy.rule_policy.rule import RulePolicy


def eval_metrics(gold_pred_sys_das):
    tp, fp, fn = 0, 0, 0
    joint_acc = total = 0
    bad_case = {}

    for dia_id, sess in gold_pred_sys_das.items():
        for turn_id, turn in sess.items():
            if not turn["gold_sys_act"] and not turn["pred_sys_act"]:
                joint_acc += 1
            elif not turn["pred_sys_act"]:
                fn += len(turn["gold_sys_act"])
            elif not turn["gold_sys_act"]:
                fp += len(turn["pred_sys_act"])
            # 当 intent 为 Recommend 或者 slot 为 周边xx 时，数据集中给出的数量并没有规律，
            # 因此，只要碰上此类，都认为正确，预测的结果基本包含数据集中的结果
            elif (
                turn["gold_sys_act"][0][0] == turn["pred_sys_act"][0][0] == "Recommend"
            ) or turn["gold_sys_act"][0][2].startswith("周边"):
                joint_acc += 1
                tp += len(turn["gold_sys_act"])
            else:
                gold = set(turn["gold_sys_act"])
                pred = set(turn["pred_sys_act"])

                if gold != pred:
                    if dia_id not in bad_case:
                        bad_case[dia_id] = {}
                    bad_case[dia_id][str(turn_id)] = {
                        "gold_sys_act": turn["gold_sys_act"],
                        "pred_sys_act": turn["pred_sys_act"],
                    }
                else:
                    joint_acc += 1

                tp += len(gold & pred)
                fn += len(gold - pred)
                fp += len(pred - gold)

            total += 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    joint_acc /= total

    output_path = os.path.join(
        get_data_path(), "crosswoz/policy_rule_single_domain_data/bad_case.json"
    )
    dump_json(bad_case, output_path)

    return f1, precision, recall, joint_acc


def main():
    rule_dst = RuleDST()
    rule_policy = RulePolicy()

    train_path = os.path.join(
        get_data_path(), "crosswoz/policy_rule_single_domain_data/train.json"
    )
    # train_path = os.path.join(get_data_path(), 'crosswoz/policy_rule_single_domain_data/single_bad_case.json')
    train_examples = json.load(open(train_path, encoding="utf8"))

    sys_state_action_pairs = {}
    for id_, dialogue in train_examples.items():
        sys_state_action_pair = {}
        sess = dialogue["messages"]
        rule_dst.init_session()
        for i, turn in enumerate(sess):
            if turn["role"] == "usr":
                rule_dst.update(usr_da=turn["dialog_act"])
                rule_dst.state["user_action"].clear()
                rule_dst.state["user_action"].extend(turn["dialog_act"])
                if i + 2 == len(sess):
                    rule_dst.state["terminated"] = True
            else:
                for domain, svs in turn["sys_state"].items():
                    for slot, value in svs.items():
                        if (
                            slot != "selectedResults"
                            and not rule_dst.state["belief_state"][domain][slot]
                        ):
                            rule_dst.state["belief_state"][domain][slot] = value

                pred_sys_act = rule_policy.predict(rule_dst.state)
                sys_state_action_pair[str(i)] = {
                    "gold_sys_act": [tuple(act) for act in turn["dialog_act"]],
                    "pred_sys_act": [tuple(act) for act in pred_sys_act],
                }
                rule_dst.state["system_action"].clear()
                rule_dst.state["system_action"].extend(turn["dialog_act"])

        sys_state_action_pairs[id_] = sys_state_action_pair

    f1, precision, recall, joint_acc = eval_metrics(sys_state_action_pairs)
    print(
        f"f1: {f1:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, joint_acc: {joint_acc:.3f}"
    )
    # f1: 0.628, precision: 0.556, recall: 0.721, joint_acc: 0.647


if __name__ == "__main__":
    main()

"""
存在的问题：
1. 数据库搜索得出多个符合约束的答案，但是数据集对应的结果有时候推荐多个，
   有时候只取一个，没有规律
2. 当约束未满足，但仍有一些符合大部分约束的结果被重新推荐，针对这个问题写过规则，
   但是去掉这个规则，指标却更高，大概是规则还需完善
3. 以周边开始的 slot 也存在数量不确定问题
4. 数据集本身的问题，比如 dialogue act 没有正确对应 utterance，比如 11184 对话
5. 门票问题，有时候 0 就是 0，不用改写成 免费
"""
