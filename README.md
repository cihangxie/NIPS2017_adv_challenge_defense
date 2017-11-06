# NIPS_2017_adv_defense_challenge
Utilize randomization to defend adversarial examples. 
Challenge URL: https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack

## The approach
The main ideal of the defense is to utilize randomization to defend adversarial examples:
- Resizing: after pre-processing, resize the original image (size of 299 x 299 x 3) to a larger size, Rnd x Rnd x 3,  randomly, where Rnd is within the range [310, 331). 
- Padding: after resizing, pad the resized image to a new image with size 331 x 331 x 3, where the padding size at left, right, upper, bottom are [a, 331-Rnd-a, b, 331-Rnd-b]. The possible padding pattern for the size Rnd is (331-Rnd+1)^2.

## Pros 
1. No training/finetuning is required
2. Very little computation introduced
3. Compatiable to different networks and different defending methods (i.e., we use randomization + ensemble adversarial training + Inception-Resnet-v2 in our submission)

## Ensemble adversarial training model
- http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz

## Team Member
- Cihang Xie (Johns Hopkins University)
- Zhishuai Zhang (Johns Hopkins University)
- Jianyu Wang (Baidu Research)
- Zhou Ren (Snap Inc.)

## Tech Report
Mitigating adversarial effects through randomization (https://drive.google.com/open?id=1fqcMZA-kC1edHp0mRXeK_0lmvcvbN5DN)
