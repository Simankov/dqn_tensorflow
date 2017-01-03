import pygame
import pygame.surfarray
import Database
import tensorflow as tf
import threading
import numpy as np
import time
import random
import copy
import os
import filecmp
from pong import Pong
from PIL import Image

from Queue import Queue
last_score_1 = 0
last_score_2 = 0
gamma = 0.9
qaction_values = []
paused = False
switch = copy.copy
queue_database_task = Queue()

#debug flags
is_fully_random = False
is_run_photos = False
is_run_training = True
is_save_in_database = True
is_run_train_without_thread = True
is_run_game = True
is_run_dqn_algo = True
is_cheat = False
is_test_dqn = True

#parameters 
WIDTH = 72  # must has delimiter 24
HEIGHT = 72  # must has delimiter 24
FRAME_COUNT = 4
NUMBER_OF_ACTIONS = 3
FULLY_CONNECTED_SIZE = (WIDTH / 8) * (HEIGHT / 8) * 32
TRAIN_WITHOUT_THREAD_COUNT = 10
STD_DEV = 0.0001
TOTAL_FRAMES_BEFORE_FULL_Q = 100000.0

maximum_time = 0


def worker():
    while True:
        task = queue_database_task.get()
        func = task[0]
        args = task[1:]
        func(*args)
        queue_database_task.task_done()


class PongAgent:

    def __init__(self):
        self.prev_variables = {}
        self.pong = Pong()
        self.random = np.random.seed(377)
        self.last_score_1 = 0
        self.last_score_2 = 0
        self.total_count = 0
        self.count_states = 0
        self.is_new_state = True
        self.time = 0
        self.round_id = 0
        self.database = Database.Database()
        self.isAfterState = False
        self.action = "nothing"
        self.reward = None
        self.is_action_available = True
        self.is_action_changed = False

        self.availableActions = dict(down=pygame.event.Event(pygame.KEYDOWN, {"key": 274}),
                                     up=pygame.event.Event(pygame.KEYDOWN, {"key": 273}), nothing=pygame.event.Event(pygame.KEYUP,{"key": 274}))

        self.prev_state = []
        self.current_state = []
        # set our on_screen_update function to always get called whenever the screen updated
        pygame.display.update = self.function_combine(pygame.display.update, self.on_screen_update)
        # FYI the screen can also be modified via flip, so this might be needed for some games
        pygame.display.flip = self.function_combine(pygame.display.flip, self.on_screen_update)
        pygame.event.get = self.function_intercept(pygame.event.get, self.makeAction)

        with tf.device("cpu:0"):

            self.inputQ, self.outputQ, self.filter_summary = self.createNetwork()
            self._session = tf.Session()
            self._action = tf.placeholder("float", [None, NUMBER_OF_ACTIONS])
            self._target = tf.placeholder("float", [None])
            readout_action = tf.reduce_sum(tf.mul(self.outputQ, self._action), reduction_indices=1)

            cost = tf.reduce_mean(tf.square(self._target - readout_action))
            self._train_operation = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
            self._session.run(tf.initialize_all_variables())
        # self.round_id = self.generate_round_id()
        if is_run_dqn_algo: threading.Thread(target=self.dqn_algo).start()
        summary_writer = tf.summary.FileWriter('/tensorflawed/tmp/logs', self._session.graph)
        if is_run_train_without_thread:
            if not os.path.exists(r'/home/sergey/dqnfiles'):
                os.mkdir(r'/home/sergey/dqnfiles')
            os.chdir(r'/home/sergey/dqnfiles')
            for v in tf.trainable_variables():
                self.prev_variables = {v.name: [] for v in tf.trainable_variables()}
            log_file = open("logfile", 'w+')
            for j in range(TRAIN_WITHOUT_THREAD_COUNT):
                for v in tf.trainable_variables():
                    filename = str(np.array(v.name.split(":"))[0]) + "_before_"+str(j)+"_iteration"
                    if len(filename) == 0: continue
                    file = open(filename, 'w+')
                    file.write(str(self._session.run(v)))
                    file.close()
                    if (j != 0) :
                        prev_filename = self.prev_variables[v.name][-1]
                        if not filecmp.cmp(filename,prev_filename):
                            log_file.write("files not same"+ filename+" "+prev_filename+'\n')
                            log_file.flush()
                    self.prev_variables[v.name].append(filename)
                self.train_dqn()
                summary = self._session.run(tf.merge_all_summaries())
            log_file.close()

        if is_run_photos: threading.Thread(target = self.getSamples).start()
        print FULLY_CONNECTED_SIZE
        if is_save_in_database: threading.Thread(target = worker).start()
        if is_run_training: threading.Thread(target = self.worker_train_dqn).start()
        if is_run_game: self.start()
        if is_test_dqn: self.test_dqn()
        return

    def start(self):
        self.pong.start()

    def worker_train_dqn(self):
        while True:
            self.train_dqn()

    def on_screen_update(self):
        self.get_feedback()
        scaled_surface = pygame.transform.scale(pygame.display.get_surface(), [WIDTH, HEIGHT])
        scaled = pygame.surfarray.array3d(scaled_surface)
        col = Image.fromarray(scaled)
        gray = col.convert('L')
        bw = gray.point(lambda x: 0 if x < 10 else 255, '1')
        image = np.asarray(bw,dtype=int)
        if self.is_action_available : return
        if self.is_action_changed:
            self.count_states += 1
            self.total_count += 1
            self.current_state.append(image)
            if (self.count_states % (FRAME_COUNT)) == 0  or self.pong.is_terminated():
                self.is_action_changed = False
                self.count_states = 0
                self.is_action_available = True
                if (len(self.prev_state) != 0):
                    before_state = self.prev_state
                else:
                    before_state = self.current_state
                after_state = self.current_state
                # fix terminal states
                l = len(after_state)
                if (l < FRAME_COUNT):
                    last = after_state[l-1]
                    for i in range(FRAME_COUNT - l):
                        after_state.append(last)

                self.loadInDatabase(before_state,
                                    after_state,
                                    self.action,
                                    0,
                                    self.round_id)
                self.prev_state = copy.deepcopy(after_state)
                self.current_state = []

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
        for key in self.availableActions.keys():
            sample = filter(lambda x: x[1]==key,samples)[0]
            action = sample[1]
            print "get_samples:",action
            for i in range(0,FRAME_COUNT):

                state_before_image_i = Image.fromarray(sample[0][i], mode="1")
                # state_before_image_i.show("Before action "+ str(action) +str(i))
                state_before_image_i.save(key + "/before_" + str(i) + ".bmp")
            for j in range(0,FRAME_COUNT):
                state_after_image_i = Image.fromarray(sample[2][j], mode="1")
                # state_after_image_i.show("After action"+ str(action)+str(j))
                state_after_image_i.save(key + "/after_" + str(j) + ".bmp")

    def printed(x):
        print x;

    def dqn_algo(self):
        while True:
           if self.count_states < 10000000 and self.is_action_available:
                actionIndex = 0
                eps = self.e()
                if np.random.choice(a=[True,False],p=[eps,1-eps]) or is_fully_random:
                    actionIndex = random.randint(0,2)
                else:
                    if len(self.prev_state) == FRAME_COUNT:
                        self.pong.switch_pause()
                        thread = threading.Thread(target=lambda: self.Q(np.array([self.prev_state]))) # to do save in global variable very bad
                        thread.start()
                        thread.join()

                        print "Q-values:"
                        print "up:",
                        print "{:10.20f}".format(qaction_values[0][0])
                        print "nothing:",
                        print "{:10.20f}".format(qaction_values[0][1])
                        print "down:",
                        print "{:10.20f}".format(qaction_values[0][2])

                        actionIndex = np.array(qaction_values).argmax()
                        if (actionIndex == 0):
                            print "Q: up"
                        elif ((actionIndex == 1)):
                            print "Q: nothing"
                        else:
                            print "Q: down"
                        self.pong.switch_pause()
                    else:
                        print "failed : length of current_state is not FRAME_COUNT"
                        print FRAME_COUNT - len(self.current_state)
                if (actionIndex == 0):
                    self.action = "up"
                elif ((actionIndex == 1)):
                    self.action = "nothing"
                else:
                    self.action = "down"
                self.is_action_available = False
                self.is_action_changed = True


    def timer_dqn_algo(self):
        threading.Thread(target=self.dqn_algo).start()

    def Q(self, states):
        global qaction_values
        time1 = time.time()
        with tf.device("cpu:0"):
            cheat_code = "none"
            if is_cheat : cheat_code = "0"
            shaped = np.rollaxis(states,1,4,cheat_code)
            qvalue = self._session.run(self.outputQ, feed_dict={self.inputQ: shaped})
            time2 = time.time()
            # print ("time of Q function with batch size " + str(shaped.shape[0]))
            # print (time2 - time1)
            if shaped.shape[0] == 1 : qaction_values = qvalue
            return qvalue

    def generate_round_id(self):
        unique = set()
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        return "".join(random.choice(chars) for _ in range(40))

    def e(self):
        b = 1.0
        a = (0.10 - b) / TOTAL_FRAMES_BEFORE_FULL_Q
        probability = a*self.total_count + b
        if self.total_count > TOTAL_FRAMES_BEFORE_FULL_Q: probability = 0.1
        if self.total_count % 100 == 0: print "probability", "{:10.20f}".format(probability)
        return probability

    def get_states(self):
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
                actions.append([0, 1, 0]);
            if (tuple[1] == "up"):
                actions.append([1, 0, 0]);
            if (tuple[1] == "down"):
                actions.append([0, 0, 1]);
            after_states.append(tuple[2])
            rewards.append(tuple[3])
        return (before_states,actions,after_states,rewards)

    def train_dqn(self):
        global maximum_time
        (before_states, actions, after_states, rewards) = self.get_states()
        time1 = time.time()
        qvalues = np.array(np.max(self.Q(np.array(after_states)),axis=1))
        time2 = time.time()
        if time2 - time1 > maximum_time :
            maximum_time = time2 - time1
        print maximum_time
        targets = rewards + qvalues*gamma
        time1 = time.time()
        cheat_code = "none"
        if is_cheat: cheat_code = "1"
        self._session.run(self._train_operation,feed_dict={self._target:targets,self._action:actions,self.inputQ:np.rollaxis(np.array(before_states),1,4,cheat_code)})
        time2 = time.time()
        print ("time of training")
        print (time2 - time1)

    def function_intercept(self,intercepted_func, intercepting_func):
        def wrap(*args, **kwargs):
         # call the function we are intercepting and get it's result
         real_results = intercepted_func(*args, **kwargs)
         # call our own function and return our new results
         new_results = intercepting_func(real_results, *args, **kwargs)
         return new_results
        return wrap

    def makeAction(self, actual_events, *args, **kwargs):
        return [self.availableActions.get(self.action)]

    def test_dqn(self):
        log_file = open("log_chech_q", "w+")
        prev_value = []
        i = 0
        (before_states, actions, after_states, rewards) = self.get_states()
        for before_state in before_states:
            i += 1
            value = self.Q(np.array([before_state]))
            if len(prev_value) > 0:
                if not np.array_equal(value, prev_value):
                    print "value:", value
                    print "not_equal ", str(i)
                    log_file.write("not equal" + str(i))
                    log_file.flush()
            prev_value = value
        log_file.close()


    def get_feedback(self):
        global last_score_1,last_score_2
        scores = self.pong.get_scores()
        bar1_score = scores[0]
        bar2_score = scores[1]
        score_change = (bar1_score - last_score_1) - (bar2_score - last_score_2)
        last_score_2 = bar2_score
        last_score_1 = bar1_score
        if (abs(score_change) > 0):
            print "SCORE ",bar1_score," ",bar2_score
            if (score_change > 0):
                 queue_database_task.put((Database.Database.setRewards,copy.copy(self.round_id),True))
                 # print "set +1", self.round_id
            else :
                 queue_database_task.put((Database.Database.setRewards,copy.copy(self.round_id),False))
                 # print "set -1", self.round_id

            self.round_id = self.generate_round_id()

        return score_change

    def createNetwork(self):
        conv_layer_1_biases = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[16]),name="cl1b")
        conv_layer_1_weights = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[8,8,FRAME_COUNT,16]),name="cl1w")
        input_layer = tf.placeholder("float", [None,WIDTH,HEIGHT,FRAME_COUNT])
        conv_layer_1 = tf.nn.relu(tf.nn.conv2d(input_layer, strides=[1,4,4,1], filter=conv_layer_1_weights, padding = 'SAME')  + conv_layer_1_biases)

        # max_pool_layer = tf.nn.max_pool(conv_layer_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        conv_layer_2_biases = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[32]),name="cl2b")
        conv_layer_2_weights = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[4, 4, 16, 32]),name="cl2w")
        conv_layer_2 = tf.nn.relu(tf.nn.conv2d(conv_layer_1, strides=[1,2,2,1],filter=conv_layer_2_weights, padding = 'SAME') + conv_layer_2_biases)

        reshaped_layer = tf.reshape(conv_layer_2,[-1,FULLY_CONNECTED_SIZE])

        fully_connected_layer_weights = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[FULLY_CONNECTED_SIZE,256]),name="fclw")
        fully_connected_layer_biases = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[256]),name="fclb")

        fully_connected_layer = tf.nn.relu(tf.matmul(reshaped_layer,fully_connected_layer_weights) + fully_connected_layer_biases)

        output_layer_weights = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[256,NUMBER_OF_ACTIONS]),name="olw")
        output_layer_biases = tf.Variable(tf.truncated_normal(stddev=STD_DEV, shape=[NUMBER_OF_ACTIONS]),name="olb")

        output_layer = tf.matmul(fully_connected_layer,output_layer_weights) + output_layer_biases
        filter_summary = tf.summary.tensor_summary("summary",conv_layer_1_biases)
                                # conv_layer_1_weights,
                                # conv_layer_2_biases,
                                # conv_layer_2_weights,
                                # fully_connected_layer_biases,
                                # fully_connected_layer_weights,
                                # output_layer_weights,
                                # output_layer_biases])

        return input_layer, output_layer, filter_summary



