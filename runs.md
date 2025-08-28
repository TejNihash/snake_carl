so first we trained and saved it in q_net_mark1_dash.pth

next up, we loaded that and saved in q_net_mark2.pth using 200 episodes and 0.8 epsilon start, and mouses decaying and 100 steps per episode.done 

next up, lets have 200 episodes, 0.8 epsilon, 200 steps and reward of -0.1 for every useless move(hoping it will learn to attack quickly). don't forget to add it to the total reward. 


and then do it again as we did previously.

next up, load the same model but put in the maze. 