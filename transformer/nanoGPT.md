# nanoGPT

## prepare env

```bash

# check gpu
update-pciids
lspci -v | less
# NVIDIA Corporation GA100 [A100 SXM4 80GB]

# install cuda
nvcc --version
# 11.4

# preprae env
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-2023.03-Linux-x86_64.sh
conda create -n nanoGPT python=3.9
source activate nanoGPT

# install
pip install torch --extra-index-url https://download.pytorch.org/whl/cu114
pip install transformers datasets tiktoken wandb numpy
#might try 11.3 also because 11.4 works with 11.3

```

## train

This is a mini version of gpt training from scratch. Using characters from shakespeare dataset

```bash
# train shakespare & sample
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare_char/input.txt
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
#iter=5000

# inference
python sample.py --out_dir=out-shakespeare-char
```

Inference results example

```text
YORK:
I think you, to look to be a table,
Or take no triumphant here and her wife,
And from her ane army with her daughter's daughter.

YORK:
When stand the beggins, son, in this manners to the wind,
And in warm well-and tears thus would by stand all the tyrants
I'll not patiently home: I am mother
To Warwick that is so out of much a gentleman poor
As he is here disterds affected of the flowers.

Post:
Here Plantagenet, and Plantagenet,
Have disdaining here heir courts.

QUEEN MARGARET:
But why
---------------

Men part, I must to be poor many of a child,
And then the meals and show the heaven like body.
I'll not speak to-day; lords, instruction and sorted
Perbut movet: if it be through thee to the sting,
As thou Richard stand'st not sucking to look.

DUKE OF AUMERLE:
Heaven as he did, what say thou seest,
And stop his victory, as so I surply,
That by the noison of the rest thou art underneath?
The summons protectors of that thou sayst,
Being king casting me with me wrong the own
With flatter than thee
---------------

Men take me with all, I have no more.
My most gracious lord, of any moans,
I have done, my wife, and will we have deceived,
Or my tongue did chance answer to wash.I
KING LEWIS XI:
What, is that Plantagenet more than this side,
And not my hands of this antig detecty?

WARWICK:
Now my lord, go, my lord; what are the world murdered.
Ah, that my name hath done, but my true love!
And have been so much length to you,
Whose boer who hath the depended moved from your love,
And so in yours are will have
---------------
```

## fine tune

A common use of large gpt base model is to fine tune it based on a domain dataset.

```bash
# fine tune shakespare
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare/input.txt
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
# iter=20
python sample.py --out_dir=out-shakespeare
```

Sampling is much slower but better? in this case.

```text

MERYLA:
They are my sisters:
For the father's death, we have fled.

CAMILLO:
Live, and I will follow thee.

MERYLA:
I do not love thee, Camillo; nor yet
Camellia.

CAMILLO:
If I did, thou wouldst return, and I
would not follow thee.

MERYLA:
I am not so mad as thou art mad.

CAMILLO:
Thy father, he who was my father,
Doth live and I do live: why dost thou stay?

MERYLA:
I do stay: that thou shouldst follow me is unright.

CAMILLO:
Unright! I am come to hunt him.

MERYLA:
Do not, thou shalt not kill him:
He is a good man, and an excellent soldier.

MORGAN:
Is he so, my Lord?

MERYLA:
More than good: he is more than a soldier.
He hath in him more virtue, than warlike bravery:
And when the Lord's name is called, he yields
The soldier's right to his lips:
But when it is said, 'Kill him,' he looks like a man
That would rather starve than do battle.
Look thou, my lord, how he shakes his sword
To drive off the foe: he that speaks it,
Bends his knee, and kneels to the earth:
As he doth, the poor helmsman groaneth:
But when the Lord's name is spake, the man
Keeps his place, and raises his knee to heaven.

DUKE OF AUMERLE:
What a change is this!
One moment he is a soldier, to another
A man that hath no virtue or any beauty,
But a vile, mean, repulsive, and cruel man.

MORGAN:
And yet thou wouldst not die, thou wouldst live.

DUKE OF AUMERLE:
Dost thou know that I would not?

MORGAN:
True, I have seen thy manner.

DUKE OF AUMERLE:
I am a man of good
---------------
```

## inference

Infer from .

```bash
python sample.py \
 --init_from=gpt2-xl \
 --start="I want you to act as an artist advisor providing advice on various art styles such tips on utilizing light & shadow effects effectively in painting, shading techniques while sculpting etc." \
 --num_samples=1
 --max_new_token=1000

```

Result is

```text
I want you to act as an artist advisor providing advice on various art styles such tips on utilizing light & shadow effects effectively in painting, shading techniques while sculpting etc. I want you to share your techniques & resources with us so we can apply it to our own practices.

If you are accepted as an intern by me, you will have an opportunity to attend a two week residency training program in my studio and then be admitted to the program as an artist-intern. Interns will have the opportunity to meet with senior artists and discuss their progress throughout the program. In addition, I will pay you a stipend and provide you with a room in my studio for the duration of your internship.

The internship program is for artists who are 18 years old and above.

If accepted, an artist will be selected for this internship program based on my criteria for selection. The only requirement I have for you is that you have at least 3 years of professional experience in an art related field. You must be creative and have exceptional artistic talent. You must have 1-2 years experience in web development, design & graphic design, graphic design, illustration, printmaking, graphic design, photography, web design and/or web development.

Some artists have some knowledge of photography or web design, but for the most part, the intern will work directly with me in my studio. If you are interested in being a part of this program, please apply for the unpaid Intern position HERE.

If you are interested in applying or getting involved with my studio, please send your resume to me at: info@serendipityart.com or call (831) 829-7905.

Thank you for your consideration and interest in my internship program. If interested, please go to: Serendipity Art Intern – Apply Now<|endoftext|>Share Pinterest

Email

The Porsche 919 hybrid hypercar, which will compete in the FIA World Endurance Championship, will be on display at the Paris Motor Show.

The car, revealed by Porsche in March, is an electric sports car that combines a mid-engine, rear-drive architecture with a Formula 1-style hybrid powertrain. Its aim is to have a driving range of at least 600 miles and be capable of returning a top speed of more than 210 mph.

We first saw a prototype at the Nürburgring last week, which it said would be ready in May. It's expected to be rolled out to the public in the summer.

The hybrid powertrain uses two electric motors to run the rear wheels and a third to drive
```
