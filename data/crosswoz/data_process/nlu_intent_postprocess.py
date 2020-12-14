import re


def is_slot_da(da):
    tag_da = {"Inform", "Recommend"}
    not_tag_slot = "酒店设施"
    if da[0] in tag_da and not_tag_slot not in da[2]:
        return True
    return False


def calculate_f1(predict_golden):
    tp, fp, fn = 0, 0, 0
    for item in predict_golden:
        predicts = item["predict"]
        labels = item["golden"]
        for ele in predicts:
            if ele in labels:
                tp += 1
            else:
                fp += 1
        for ele in labels:
            if ele not in predicts:
                fn += 1
    precision = 1.0 * tp / (tp + fp) if tp + fp else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def tag2das(word_seq, tag_seq):
    assert len(word_seq) == len(tag_seq)
    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith("B"):
            intent, domain, slot = tag[2:].split("+")
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith("I") and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith("##"):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            das.append([intent, domain, slot, value])
        i += 1
    return das


def intent2das(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, domain, slot, value = re.split(r"\+", intent)
        triples.append([intent, domain, slot, value])
    return triples


def recover_intent(dataloader, intent_logits):
    das = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intent_domain = dataloader.id2intent[j]
            das.append(intent_domain)

    return das
