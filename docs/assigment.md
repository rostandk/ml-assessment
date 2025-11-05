ML Scientist Technical Take-home Task
Email Instructions
You will be working on a machine learning training pipeline initially developed by a junior
data science colleague. Your task is to take over the project, improve the model, and present
your approach and reasoning.
The goal of this exercise is to understand how you approach a real-world ML project —
including identifying weaknesses, prioritising improvements, and designing robust solutions.
This is not a fixed-question test; instead, you’ll be assessed on the quality of your judgment,
the improvements you make, and your ability to communicate them clearly.
We are flexible with timing – we recommend that you don’t spend more than 4 hours to
complete the task.
Once you confirm a time that works for you, we’ll send you:
●
A project brief (in README.md)
●
A starting notebook
●
Supporting data and files
Please reply with your preferred time, and we’ll send everything through then.
Task Instructions
(duplicated from README.md)
# Hello!
Thank you for taking the time to do Depop’s coding test 🙏.
This project aims to catch keyword spamming 🍖 — when sellers add large numbers of
unrelated or irrelevant keywords in an item’s description to boost its ranking in search results
📈
. For example, a buyer searching for “Levi jeans”
👖 might see Diesel jeans ⛽ ranked
highly, simply because the word “Levi” was spammed in the description. This is frustrating for
the buyer 😖
.
Here are examples of product descriptions with keyword spam:
```
Low waist/rise diesel bootcut/flared jeans. Size XS/6. Great condition. Cool red stitching
details.
Message for any questions :) UK shipping only
No returns
#vintage #diesel #denim #lowrise #levi #wrangler #lee #y2k #90s #2010s #blue #black
#faded
```
```
Low rise y2k blue Diesel bootcut jeans
Size label W29 L32
Flat laid measurements below —
32 inch waist (sits on hips)
7 inch rise
32 inch inseam
FREE UK SHIP
£15 international
Ignore: 80s 90s y2k baggy navy jeans denim levi calvin klein
```
If we can classify item descriptions as ‘spammy’ 🍖, we can demote those items in the
ranking algorithm 📉. This project is focused on building that classifier 🔨.
This test imagines you’re working alongside a junior data scientist who has already put
together an initial model. They’ve shared a notebook with their code. Normally, you might
review and give feedback to coach them - but in this case, you should take over the work
and bring it up to your high standards 🏆
.
Your goal is to identify flaws in the current approach and fix the most impactful issues - within
the time you can allocate to the task.
As you’ll see, there are a bunch of issues:
•
Some errors in ML logic 🔬
•
Badly tuned model
•
Limited features
•
Lack of structure 🏗 (likely because it’s all in a notebook)
•
Un-Pythonic patterns
•
And generally, it needs a lot of improvement 🛠
There’s too much here to perfect everything in one go, so focus on the improvements that
matter most. At the end, let us know what you’d do next if you had more time ⏰, and
suggest logical next steps for the project 󰩢.
After you submit your work, we’ll schedule a follow-up interview where you can walk us
through your thinking and decisions.
NB: The code has been tested with Python 3.11, but feel free to use any Python 3 version
you prefer.



Thank you for taking the time to do Depop’s coding test 🙏.

This project aims to catch keyword spamming 🍖 — when sellers list large numbers of unrelated or irrelevant keywords in an item’s description to boost its ranking in search results 📈. For example, a buyer searching for “Levi jeans” 👖 might see Diesel jeans ⛽️ ranked highly, simply because the word “Levi” was spammed in the description. This is frustrating for the buyer 😖.

Here are examples of product descriptions with keyword spam:

```
Low waist/rise diesel bootcut/flared jeans. Size XS/6. Great condition. Cool red stitching details.
Message for any questions :) UK shipping only
No returns
#vintage #diesel #denim #lowrise #levi #wrangler #lee #y2k #90s #2010s #blue #black #faded
```

```
Low rise y2k blue Diesel bootcut jeans  
Size label W29 L32  
Flat laid measurements below —  
32 inch waist (sits on hips)  
7 inch rise  
32 inch inseam  
FREE UK SHIP  
£15 international  
Ignore: 80s 90s y2k baggy navy jeans denim levi calvin klein
```

If we can classify item descriptions as ‘spammy’ 🍖, we can demote those items in the ranking algorithm 📉. This project is focused on building that classifier 🔨.

This test imagines you’re working alongside a junior data scientist who has already put together an initial model. They’ve shared a notebook with their code. Normally, you might review and give feedback to coach them - but in this case, you should take over the work and bring it up to your high standards 🏆.

Your goal is to identify flaws in the current approach and fix the most impactful issues - within the time you can allocate to the task.

As you’ll see, there are a bunch of issues:
	•	Some errors in ML logic 🔬
	•	Badly tuned model
	•	Limited features
	•	Lack of structure 🏗 (likely because it’s all in a notebook)
	•	Un-Pythonic patterns
	•	And generally, it needs a lot of improvement 🛠

There’s too much here to perfect everything in one go, so focus on the improvements that matter most. At the end, let us know what you’d do next if you had more time ⏰, and suggest logical next steps for the project 🧗🏿‍♀️.

After you submit your work, we’ll schedule a follow-up interview where you can walk us through your thinking and decisions.

NB: The code has been tested with Python 3.11, but feel free to use any Python 3 version you prefer.


-------requirements.txt---------
pandas==2.2.3
numpy==2.2.6
scikit-learn==1.6.1
spacy==3.8.6
truecase==0.0.14
xgboost==3.0.1
unidecode==1.4.0


