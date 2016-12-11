import pygame
import pygame.surfarray
import Database
from Paused import switch_pause
import tensorflow as tf
import threading
import numpy as np
import time
import random
import copy
from  pong import Pong
from PIL import Image
# from pong import switch_pause

from Queue import Queue
last_score_1 = 0
last_score_2 = 0
gamma = 0.9
e = 1.0
qaction_values = []
paused = False
switch = copy.copy
queue_database_task = Queue()

def worker():
    while True:
        task = queue_database_task.get()
        func = task[0]
        args = task[1:]
        func(*args)
        queue_database_task.task_done()


class PongAgent:

    def __init__(self):
        self.pong = Pong()
        self.random = np.random.seed(377)
        self.last_score_1 = 0
        self.last_score_2 = 0
        self.screen_counter = 1
        self.count_states = 0
        self.is_new_state = True
        self.WIDTH = 72  # must has delimiter 24
        self.HEIGHT = 72 # must has delimiter 24
        self.fully_connected_size = (self.WIDTH / 8) * (self.HEIGHT / 8) * 32
        self.time = 0
        self.round_id = 0
        self.database = Database.Database()
        self.FRAME_COUNT = 4
        self.NUMBER_OF_ACTIONS = 3
        self.isAfterState = False
        self.action = "nothing"
        self.reward = None
        self.is_action_available = False
        self.lower_limit_count_state_to_db = 10
        self.is_exploration = False
        self.is_best_response = False
        self.availableActions = dict(down=pygame.event.Event(pygame.KEYDOWN, {"key": 274}),
                                     up=pygame.event.Event(pygame.KEYDOWN, {"key": 273}), nothing=pygame.event.Event(pygame.KEYUP,{"key": 274}))

        self.array_of_frames = []
        self.array_of_frames_after_state = []
        self.current_state = []
        # set our on_screen_update function to always get called whenever the screen updated
        pygame.display.update = self.function_combine(pygame.display.update, self.on_screen_update)
        # FYI the screen can also be modified via flip, so this might be needed for some games
        pygame.display.flip = self.function_combine(pygame.display.flip, self.on_screen_update)
        pygame.event.get = self.function_intercept(pygame.event.get, self.makeAction)

        with tf.device("cpu:0"):

            self.inputQ, self.outputQ = self.createNetwork()
            self._session = tf.Session()
            self._action = tf.placeholder("float", [None, self.NUMBER_OF_ACTIONS])
            self._target = tf.placeholder("float", [None])
            readout_action = tf.reduce_sum(tf.mul(self.outputQ, self._action), reduction_indices=1)

            cost = tf.reduce_mean(tf.square(self._target - readout_action))
            self._train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)
            self._session.run(tf.initialize_all_variables())
        # self.round_id = self.generate_round_id()
        self.timer_dqn_algo()
        # threading.Thread(self.getSamples()).start()
        print self.fully_connected_size
        # for i in range(0,40):
        #    time1 = time.time()
        #    self.train_dqn()
        #    print (time.time() - time1)
        #    time2 = time.time()
        #    print self.Q(np.array([Database.Database.getSamples()[0][0]]))
        #    print time2 - time.time()
        threading.Thread(target = worker).start()
        for j in range(6):
            threading.Thread(target = self.worker_train_dqn).start()
        self.start()
        return

    def start(self):
        self.pong.start()

    def epsilon(self):
        maximum = 1000000;


    def worker_train_dqn(self):
        while True:
            self.train_dqn()
            time.sleep(2)

    def on_screen_update(self):
        self.get_feedback()
        scaled_surface = pygame.transform.scale(pygame.display.get_surface(), [self.WIDTH, self.HEIGHT])
        scaled = pygame.surfarray.array3d(scaled_surface)
        col = Image.fromarray(scaled)
        gray = col.convert('L')
        bw = gray.point(lambda x: 0 if x < 10 else 255, '1')
        image = np.asarray(bw)
        if self.is_action_available: return
        if (self.isAfterState) :
            self.array_of_frames_after_state.append(image)
        else:
            self.array_of_frames.append(image)
        if (self.screen_counter % (self.FRAME_COUNT + 1)) == 0:
            self.count_states += 1
            self.is_new_state = True
            if self.isAfterState:
                self.screen_counter = 1
                assert self.array_of_frames is not None
                assert self.array_of_frames_after_state is not None
                assert self.action is not None
                assert self.reward is not None

                if (self.count_states > self.lower_limit_count_state_to_db):
                    self.loadInDatabase(self.array_of_frames,
                                        self.array_of_frames_after_state,
                                        self.action,
                                        self.reward,
                                        self.round_id)
                self.current_state = np.array(self.array_of_frames_after_state)
                self.array_of_frames_after_state = []
                self.array_of_frames = []
                self.reward = 0
                self.isAfterState = False
                self.is_action_available = False
            else:
                self.screen_counter = 1
                self.isAfterState = True
                self.reward = 0
                self.is_action_available = True
        self.screen_counter += 1

    def function_combine(self,screen_update_func, our_intercepting_func):
        def wrap(*args, **kwargs):
         screen_update_func(*args, **kwargs)  # call the screen update func we intercepted so the screen buffer is updated
         our_intercepting_func()  # call our own function to get the screen buffer

        return wrap

    def loadInDatabase(self, state_before, state_after, action,reward,round_id):
        # threading.Thread(target=self.database.add, args=(state_before,action,state_after,
        #                  reward,round_id)).start()
        # self.database.add(state_before=state_before,action=action,state_after=state_after, reward=reward,round_id=round_id)
        queue_database_task.put((self.database.add,copy.copy(state_before),copy.deepcopy(action),copy.copy(state_after),copy.copy(reward),copy.copy(round_id)))

    def getSamples(self):
        samples = self.database.getSamples()
        if (len(samples) == 0): return
        sample = filter(lambda x: x[1]=="nothing",samples)[0]
        action = sample[1]
        print "get_samples:",action
        for i in range(0,4):
            state_before_image_i = Image.fromarray(sample[0][i],mode="1")
            state_before_image_i.show("Before action "+ str(action) +str(i))
            time.sleep(5)
        for j in range(0,4):
            state_after_image_i = Image.fromarray(sample[2][j],mode="1")
            state_after_image_i.show("After action"+ str(action)+str(j))
            time.sleep(5)


    def dqn_algo(self):
        while True:
           if self.count_states < 10000000 and self.is_action_available:
                epsilon = 0.5
                actionIndex = 0
            # self.is_exploration = np.random.choice([False,True], p=[1 - epsilon, epsilon])
                self.is_exploration = random.randint(0,4) == 1
                self.is_best_response = not self.is_exploration
                # if self.is_exploration:
                if random.randint(0,self.e()) == 0:
                    actionIndex = random.randint(0,2)
                else:
                    if len(self.current_state) != 0:
                        print "begin"
                        self.pong.switch_pause()
                        thread = threading.Thread(target=lambda: self.Q(np.array([self.current_state]))) # to do save in global variable very bad
                        thread.start()
                        thread.join()
                        time.sleep(5)
                        print qaction_values
                        actionIndex = np.array(qaction_values).argmax()
                        self.pong.switch_pause()
                    else:
                        print "failed : length of current_state is 0"
                if (actionIndex == 0):
                    self.action = "up"
                elif ((actionIndex == 1)):
                    self.action = "nothing"
                else:
                    self.action = "down"
                print self.action
                self.is_action_available = False

    def timer_dqn_algo(self):
        threading.Thread(target=self.dqn_algo).start()

    def Q(self, states):
        global qaction_values
        with tf.device("cpu:0"):
            shaped = np.rollaxis(states,1,4)
            qvalue = self._session.run(self.outputQ, feed_dict={self.inputQ: shaped})
            if shaped.shape[0] == 1 : qaction_values = qvalue
            return qvalue

    def generate_round_id(self):
        unique = set()
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        return "".join(random.choice(chars) for _ in range(40))

    def e(self):
        if (self.count_states > 6000):
            return 1
        else:
            return 0


    def train_dqn(self):
        batch = Database.Database.getSamples()
        if (len(batch) == 0):
            print "database is empty"
            return
        before_states = []
        rewards = []
        after_states = []
        actions = []
        for tuple in batch:
            before_states.append(tuple[0])
            if (tuple[1] == "nothing"):
                actions.append([0,1,0]);
            if (tuple[1] == "up"):
                actions.append([1,0,0]);
            if (tuple[1] == "down"):
                actions.append([0,0,1]);
            after_states.append(tuple[2])
            rewards.append(tuple[3])
        qvalues = np.array(np.max(self.Q(np.array(before_states)),axis=1))
        before_states = np.rollaxis(np.array(before_states),1,4)
        print "trained"
        targets = rewards + qvalues*gamma
        self._session.run(self._train_operation,feed_dict={self._target:targets,self._action:actions,self.inputQ:before_states})

    def function_intercept(self,intercepted_func, intercepting_func):
        def wrap(*args, **kwargs):
         # call the function we are intercepting and get it's result
         real_results = intercepted_func(*args, **kwargs)
         # call our own function and return our new results
         new_results = intercepting_func(real_results, *args, **kwargs)
         return new_results
        return wrap

    def makeAction(self, actual_events, *args, **kwargs):
        # print self.action
        # if self.action == "nothing":
        #     return []
        # else:
            return [self.availableActions.get(self.action)]

    def get_feedback(self):
        global last_score_1,last_score_2
        scores = self.pong.get_scores()
        bar1_score = scores[0]
        bar2_score = scores[1]
        score_change = (bar1_score - last_score_1) - (bar2_score - last_score_2)
        last_score_2 = bar2_score
        last_score_1 = bar1_score
        if (abs(score_change) > 0):
            print self.round_id
            if (score_change > 0):
                 queue_database_task.put((Database.Database.setRewards,copy.copy(self.round_id),True))
            else :
                 queue_database_task.put((Database.Database.setRewards,copy.copy(self.round_id),False))
            self.round_id = self.generate_round_id()

        return score_change

    def createNetwork(self):
        conv_layer_1_biases = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[16]))
        conv_layer_1_weights = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[8,8,self.FRAME_COUNT,16]))
        input_layer = tf.placeholder("float", [None,self.WIDTH,self.HEIGHT,self.FRAME_COUNT])
        conv_layer_1 = tf.nn.relu(tf.nn.conv2d(input_layer, strides=[1,4,4,1], filter=conv_layer_1_weights, padding = 'SAME')  + conv_layer_1_biases)

        # max_pool_layer = tf.nn.max_pool(conv_layer_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        conv_layer_2_biases = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[32]))
        conv_layer_2_weights = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[4, 4, 16, 32]))
        conv_layer_2 = tf.nn.relu(tf.nn.conv2d(conv_layer_1, strides=[1,2,2,1],filter=conv_layer_2_weights, padding = 'SAME') + conv_layer_2_biases)

        reshaped_layer = tf.reshape(conv_layer_2,[-1,self.fully_connected_size])

        fully_connected_layer_weights = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[self.fully_connected_size,256]))
        fully_connected_layer_biases = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[256]))

        fully_connected_layer = tf.nn.relu(tf.matmul(reshaped_layer,fully_connected_layer_weights) + fully_connected_layer_biases)

        output_layer_weights = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[256,self.NUMBER_OF_ACTIONS]))
        output_layer_biases = tf.Variable(tf.constant(float(np.random.randint(0,1)), shape=[self.NUMBER_OF_ACTIONS]))

        output_layer = tf.matmul(fully_connected_layer,output_layer_weights) + output_layer_biases

        return input_layer, output_layer



