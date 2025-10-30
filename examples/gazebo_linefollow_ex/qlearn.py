import random
import pickle
import copy


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.policy = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def save_policy(self):
        self.policy = copy.deepcopy(self.q)
    
    def getPolicy(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.policy.get((state, action), 0.0)

    def policy_chooseAction(self, state, return_q=False):
        q_values = [self.getPolicy(state, 0), self.getPolicy(state, 1), self.getPolicy(state, 2)]
            
        max_q = max(q_values) # Step 1: Find the maximum Q-value
        best_actions = [i for i, q in enumerate(q_values) if q == max_q] # Step 2: Find all indices that have the max Q-value

        return random.choice(best_actions) # Step 3: Randomly select one of the best actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        try:
            with open(filename + ".pickle", "rb") as f:
                self.q = pickle.load(f)
            print("Loaded file: {}".format(filename + ".pickle"))
        except FileNotFoundError:
            print("File not found. Starting with empty Q-table.")
            self.q = {}

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename + ".pickle", "wb") as f:
            pickle.dump(self.q, f)
        print("Wrote to file: {}".format(filename + ".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        if(random.uniform(0.0, 1.0) > self.epsilon):
            # Exploitation
            # Example Q-values for 3 actions
            q_values = [self.getQ(state, 0), self.getQ(state, 1), self.getQ(state, 2)]
            
            max_q = max(q_values) # Step 1: Find the maximum Q-value
            best_actions = [i for i, q in enumerate(q_values) if q == max_q] # Step 2: Find all indices that have the max Q-value

            return random.choice(best_actions) # Step 3: Randomly select one of the best actions
        else:
            #Exploration
            return random.choice([0, 1, 2])

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        if(self.getQ(state1, action1) == 0):
            self.q[(state1, action1)] = 0.0

        max_next_q = max(
            self.q.get((state2, 0), 0.0),
            self.q.get((state2, 1), 0.0),
            self.q.get((state2, 2), 0.0)
        )

        self.q[(state1, action1)] += self.alpha * ( (reward + self.gamma * max_next_q) - self.getQ(state1, action1) )

        # self.q[(state1,action1)] = reward
