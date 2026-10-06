# Test-set examples

Each card is one held-out user, chosen for the most extreme model score in its cell.
Quotes are verbatim excerpts from that user's own posts, picked by hand as the passages most related to
distress. Spelling is the user's; names are removed.
The decision uses the MTM + PHQ-9 score and a threshold that maximizes F1 on that fold's development users.

## Correct, high risk · true score 5 · called high risk · score 0.99 · threshold 0.525

Open, repeated talk of suicide across 933 posts.

> i think a Lot of ppl who see me constantly talking abt suicide think im joking? but honestly the reality is im not ever joking and i am willing and ready to die at a moments notice

> Mental illness is a series of questions like when have I eaten last? Why am I crying in the bathroom at 5:16 in the pm? Was I always this sad for no reason???

## Missed high risk · true score 5 · called not high risk · score 0.00 · threshold 0.525

Rare distress, buried among 189 mostly cheerful posts.

> I saw my dad have a stroke that night and I would never wish that upon my worst enemy.

> Never let yourself get so far down a hole to where you feel you can't see the top.

> Brooding, loving, and hopeless.... Lol hmmmm

## False alarm · true score 0 · called high risk · score 1.00 · threshold 0.005

Self-critical, low-mood posts, but no reported ideation.

> My kids are growing up around me and I still haven't given them the life I wanted to. I can't help but feel like I'm failing. Too little, too late.

> I truly feel like a caged bird, unable to express my potpourri of emotions and thoughts using a creative medium.

## Correct, no risk · true score 0 · called not high risk · score 0.00 · threshold 0.005

Death and feelings come up only as jokes.

> Someone just asked me how my dad was, and I said "Dead, so not great". I didn't do that right, did I?

> Sometimes I wonder if I missed the day in kindergarten when all the other kids learned how to deal with their feelings.
