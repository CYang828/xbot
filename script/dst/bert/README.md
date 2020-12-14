# Bert-DST
This method is adapted from
[A Simple but Effective BERT Model for Dialog State Tracking on Resource-Limited Systems](https://arxiv.org/pdf/1910.12995.pdf).
The paper was originally for a single-domain dataset WOZ2.0, and we made some minor changes here to apply to MultiWOZ
![bert-dst](../../../asset/bert-dst.png)


## Change Description
1. Add domain to candidate Slot-Value Pair, like **Restaurant - food = Australian**
2. There is a `ontology.json` in the [source project](https://github.com/laituan245/BERT-Dialog-State-Tracking), here
are the slot and the corresponding values, Compared with the multi-domain MultiWOZ, it is much less, so the method of
constructing the data set according to the source code is a disaster for MultiWOZ, which has about 15,000 lines. To preprocessing
data faster, We used a multi-process approach, but this step is still time-consuming, in the end we only used one-tenth of
the training set, which is about 500 dialogues.
3. The original ontology of MultiWOZ has been cleaned up, see `script/dst/bert/utils.py` for the specific processing process
4. For the action of Request type, because it has no value, using Request to replace Slot and Slot to replace Value

## Arguments Explanation
all training arguments are save in `xbot/config/dst/bert/train.json`
* random_undersampling: Whether to undersample negative samples，default: 1 (true)
* overall_undersampling_ratio: The ratio of undersampling training set, default: 0.1
* debug: Whether to use debug mode, In debug mode, you can quickly run a complete process, default: 0
* use_cache_data: Whether to use cached preprocessed data, default: 1

## Example Usage
To train a model from scratch, run following command:
```bash
$ cd xbot
$ python -m script.dst.bert.train
```

## Issues
At present, we feel that the way to calculate the joint goal in the source project is not reasonable. Here are some ideas:

If the current label has only the request type, and the belief state contains the previous inform, and the pred inform
must be empty, the pred recovered will definitely not be equal to the gold recovered. In another case, the turn label
is empty, but the belief state is not empty, and if the model is good enough, the prediction is definitely empty, then
the joint goal is definitely not equal.

example (dev.json, dialogue_id:600, last turn):
```json
{
    "system_acts":[

    ],
    "system_transcript":"Chicquito's is located at 2G Cambridge Leisure Park Cherry Hinton Road Cherry Hinton, Phone 01223 400170, and the restaurant below that is called frankie and bennys. It has a phone number of 01223 412430 and address of Cambridge Leisure Park Clifton Way Cherry Hinton",
    "num":{
        "system_acts":[
            [
                0,
                16,
                1
            ]
        ],
        "transcript":[
            0,
            34,
            35,
            15,
            35,
            36,
            37,
            38,
            39,
            15,
            1
        ]
    },
    "belief_state":[
        {
            "slots":[
                [
                    "price range",
                    "expensive"
                ]
            ],
            "act":"inform"
        },
        {
            "slots":[
                [
                    "area",
                    "south"
                ]
            ],
            "act":"inform"
        }
    ],
    "turn_id":2,
    "transcript":[
        "Thank",
        "you",
        ".",
        "You",
        "have",
        "been",
        "most",
        "helpful",
        "."
    ],
    "turn_label":[

    ]
}
```

Welcome everyone to open issue！

### new issues
1. Because of the reasons of undersampling described above, for thousands of unlabeled samples constructed in
a round of dialogue in the inference stage, so we can't undersampling them，this leads to inconsistencies in
the training and inference stages, we get a high recall but very low precision. This phenomenon also proves that the
single-domain `Dialogue State Tracking` method is difficult to apply in cross-domain dataset.
2. After the above undersampling, the ratio of positive and negative samples is 1:1, so in order to alleviate the gap
between the training phase and the inference phase, can we increase the ratio of negative samples appropriately,
for example, 1:3 (this ratio is randomly determined , you can try other values at will),
after experiments, it is found that it does have a little effect, and the accuracy is slightly improved
3. Observing bad cases, we can find that some correctly predicted values are replaced by incorrectly predicted values,
So we can group these values according to their domain-slot pair, then sort them by model output's logits, take the
value corresponding to the largest logit as the value of the current domain-slot. Meanwhile, to reduce total number
of predicted labels, here we take top 5 of predictions. The following table is the experimental results, Although we
have taken some measures, the results are still not satisfactory.

    | pos-neg-ratio (pos:neg) | ranking | turn-inform | turn-request | joint-goal |
    | :---------------------: | :-----: | :---------: | :----------: | :--------: |
    |           1:3           |    -    |    0.030    |    0.110     |   0.000    |
    |           1:5           |    -    |    0.062    |    0.138     |   0.000    |
    |           1:5           | top-10  |    0.100    |    0.263     |   0.000    |
    |           1:5           |  top-6  |    0.237    |    0.362     |   0.025    |
    |           1:5           |  top-5  |    0.275    |    0.362     |   0.025    |

4. You can give full play to your imagination to improve the model, you can refer to [TriPy](https://arxiv.org/pdf/2005.02877).
Or extract single-domain data from CrossWOZ and train a single-domain BERT-DST
