# Harvest Reputation experiments

This folder contains additional materials regarding the reputation system experiments conducted in the Harvest Open environment.
Training code, configs, and launch scripts live in the [repository root README](../README.md) and `harvest_socialjax/`.
Below you can find visualizations of multiple key dynamics observations through GIFs. These GIFs are no results on their own, but are meant to give a clearer idea of the concepts and patterns described in the research. Several key findings have been uncovered by looking closely at these GIFs. This is a selection of experimental GIFs that have provided notable insights with a short description.
Hopefully this page will enhance your understanding of the research whilst further sparking your interest into the fascinating multi-agent behaviors.

## Tier 1. Baselines

### Independent Baseline

Firstly, to properly understand how individually optimizing agents behave throughout the learning trajectory we have the following GIFs illustrating 3 key stages in the learning curve _naivety, tragedy_ and _maturity_.
These GIFs are sampled from the independent baseline seed 0 run at the following points in the reward curve to visualize the key stages of the tragedy.

![Independent baseline - reward and peace landmarks](baseline/independent_reward_peace_landmarks.png)

<details>
<summary><b>GIF peak 1</b></summary>

![Individual Baseline - peak 1](baseline/independent_peak_1.gif)

Harvesting behavior during the first 250 frames of a 1000 frame episode at the first peak. 2 agents have learned to harvest quite well, 2 have not yet. The apples in the orchard remain plentiful. Zapping is not used.
</details>

<details>
<summary><b>GIF tragedy</b></summary>

![Individual Baseline - tragedy](baseline/independent_tragedy.gif)

Harvesting behavior during the first 250 frames of a 1000 frame episode at the lowest point of the tragedy. All agents have learned to harvest effectively. The apples in the orchard are fully depleted by the end of the 250 frames (after the first 1/4 of the full episode) and will not grow back from this point. The zapping action is not applied effectively.
</details>

<details>
<summary><b>GIF recovery</b></summary>

![Individual Baseline - peak 2](baseline/independent_recovery.gif)

Harvesting behavior during the first 250 frames of a 1000 frame episode at the second peak (after the tragedy). All agents have learned to harvest and use their zapping beam often to time out their co-players. As a result, the apples in the orchard remain for longer.
</details>

<details>
<summary><b>GIF end</b></summary>

![Individual Baseline - end](baseline/independent_end.gif)

Harvesting behavior during the first 250 frames of a 1000 frame episode at the end of training (after 50k episodes). All agents have learned to harvest and use their zapping beam often to time out their co-players. As a result, the apples in the orchard remain for longer. Behavior is similar to the second peak. Agents seem to focus more on harvesting in between zapping.
</details>

### Shared Baseline

The following GIF illustrates the behavior learned through optimizing a shared reward function.
This GIF is sampled from the shared baseline seed 0 run at the final checkpoint.

<details>
<summary><b>GIF</b></summary>

![Shared Baseline - end](baseline/shared_end.gif)

Harvesting behavior during the first 250 frames of a 1000 frame episode at the end of training (after 50k episodes). The agents have learned a split effort approach. 2 agents perform the bulk of apple collection, while 2 sit (partially) at the sideline. This way the average reward is high and the agents avoid overharvesting without the need to zap each other. The apples in the orchard remain plentiful. No zapping occurs.
</details>

## Tier 2. Enforcement
The following GIFs focus on the implementation of the reputation systems. Good agents are green, Bad agents are red. Police agents are Blue with light blue zap beams (different than the yellow zap beams of regular harvesters).

### Strict enforcement
<details>
<summary><b>GIF</b></summary>

![Direct enforcement - end](enforcement/strict_enforcement.gif)

Harvesting behavior during the first 500 frames of a 1000 frame episode at the end of training (after 50k episodes) enforced strictly through a direct timeout for Bad agents. The agents follow the reputation system near perfectly. The enforcement system has to kick in only seldomly. This means agents do not zap each other and harvest efficiently at a rate that does not deplete the orchard. 
</details>

### Police enforcement
<details>
<summary><b>GIF</b></summary>

![Police enforcement - end](enforcement/police_enforcement.gif)

Harvesting behavior during the first 500 frames of a 1000 frame episode at the end of training (after 50k episodes) enforced through a learning police agent. The agents follow the reputation system quite well. As an agent becomes Bad, the police agent tracks it down and zaps it effectively. The orchard is not close to depletion, but a local patch might get barren.

Notes on police behavior. The police agent has learned to keep to the lowest part of the Grid, over viewing a large part of the field. It has effectively learned to distinguish between Good and Bad agents. It only sporadically hits a Good agent, seemingly as accidental collateral damage when targeting a Bad agent.

Note on harvesting behavior. The majority of agents seem to harvest according to the norms; even an agent that becomes Bad does not splurge but mainly continues a restrained harvesting pattern.
</details>

## Tier 3. Self-supported
The last tier aims to let agents learn sustainable harvesting through the reputation system without external enforcers.

### Peer reward
<details>
<summary><b>GIF</b></summary>

![Peer reward - end](self-supported/peer_reward.gif)

Harvesting behavior during the first 500 frames of a 1000 frame episode at the end of training (after 50k episodes) with a peer reward system. Zapping Bad agents is rewarded with 5, zapping Good punished with -5 direct signal. Good agents do not seem able to effectively target Bad agents. They get zapped themselves, or become Bad rather quickly.

</details>

### Peer reward + Good immunity
<details>
<summary><b>GIF</b></summary>

![Peer reward + immunity - end](self-supported/peer_reward_immunity.gif)

Harvesting behavior during the first 500 frames of a 1000 frame episode at the end of training (after 50k episodes) with a peer reward + immunity system. Zapping Bad agents is rewarded with 5, Good agents cannot get zapped. Good agents seem to survive longer and target Bad agents. Zapping does seem fairly random, not very efficient. Harvesting behavior does significantly improve.

</details>

## Tier 4. Tweaked learning parameters
After observing the previous Tier's peer reward GIFs, we tweaked the learning parameters for improved targeting.

### Peer reward
<details>
<summary><b>GIF at 50k episodes</b></summary>

![Peer reward - end](self-supported/peer_reward_50k.gif)

Harvesting behavior during the first 500 frames of a 1000 frame episode after 50k episodes with a peer reward system. Zapping Bad agents is rewarded with 5, zapping Good punished with -5 direct signal. Good agents are quite able to effectively target Bad agents. They do get zapped themselves, or become Bad by zapping a Good agent by accident.

</details>

<details>
<summary><b>GIF at 100k episodes</b></summary>

![Peer reward - end](self-supported/peer_reward_100k.gif)

Harvesting behavior during the first 500 frames of a 1000 frame episode after 100k episodes with a peer reward system. Zapping Bad agents is rewarded with 5, zapping Good punished with -5 direct signal. Good agents are able to effectively target Bad agents. They do get zapped themselves, or become Bad by zapping a Good agent by accident, resulting in occasional free space for Bad exploit.

</details>
