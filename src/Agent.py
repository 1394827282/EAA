import numpy as np
import pickle
import torch

from Model import LSTMNet


# 经验回放数组
class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.memory_short = []
        self.max_memory = max_memory
        self.max_memory_short = max_memory
        self.discount = discount
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def remember(self, experience):
        self.memory.append(experience)

    def remember_short(self, experience):
        self.memory_short.append(experience)

    def get_batch(self, batch_size=10):
        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = torch.linspace(1, len(self.memory), len(self.memory)).to(self.device)

            p = timerank / torch.sum(timerank.float())

            batch_idx = torch.multinomial(p, num_samples=batch_size, replacement=False)

            batch = [self.memory[idx] for idx in batch_idx.cpu().numpy()]
        else:
            batch = self.memory

        return batch

    def get_batch_short(self, threshold_value, batch_size=10):
        if len(self.memory_short) > self.max_memory_short:
            del self.memory_short[:len(self.memory_short) - self.max_memory_short]

        if batch_size < len(self.memory_short):
            timerank = torch.linspace(1, len(self.memory_short), len(self.memory_short)).to(self.device)

            p = timerank / torch.sum(timerank.float())

            batch_idx = torch.multinomial(p, num_samples=batch_size, replacement=False)

            batch = [self.memory_short[idx] for idx in batch_idx.cpu().numpy()]
        else:
            batch = self.memory_short

        batch = [(state_action, threshold_value) for (state_action, reward) in batch]

        return batch

class BaseAgent(object):
    def __init__(self, histlen):
        self.single_testcases = True
        self.train_mode = True
        self.histlen = histlen

    def get_action(self, s):
        return 0

    def get_all_actions(self, states):
        """ Returns list of actions for all states """
        return [self.get_action(s) for s in states]

    def reward(self, reward):
        pass

    def save(self, filename):
        """ Stores agent as pickled file """
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


class NetworkAgent(BaseAgent):
    def __init__(self, hidden_size, histlen, state_size, lr, lambda_reg,
                 threshold=0.95,
                 threshold_inc=0.005,
                 scale_coff=5,
                 ):
        super(NetworkAgent, self).__init__(histlen=histlen)

        self.experience_length = 10000
        self.experience_batch_size = 1000
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []
        self.episode_history_short = []
        self.iteration_counter = 0
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.lambda_reg = lambda_reg

        self.net = "lstm"
        if self.net == "lstm":
            self.model = LSTMNet(self.state_size, self.hidden_size)

        self.name = self.model.name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} GPUs {torch.cuda.get_device_name(0)}")
            self.model.to(self.device)
        else:
            print("Using CPU")

        self.model_fit = False

        self.opt = torch.optim.Adam(lr=self.lr, params=self.model.parameters())
        self.cro_loss = torch.nn.MSELoss()

        self.task_num = 0

        self.threshold = np.array([threshold] * 10)
        self.threshold_inc = np.array([threshold_inc] * 10)
        self.feature_list = []
        self.importance_list = []
        self.feature_mat = []
        self.scale_coff = scale_coff

        self.grad = 1

    def get_action(self, s):
        if self.model_fit:
            with torch.no_grad():
                inputs_tensor = torch.tensor(np.array(s).reshape(1, -1), dtype=torch.float32).to(self.device)

                outputs = self.model(inputs_tensor)
                a = outputs[0][0].item()
        else:
            a = np.random.random()
        if self.train_mode:
            self.episode_history.append((s, a))

        return a

    def setshort(self, s, a):
        if self.train_mode:
            self.episode_history_short.append((s, a))

    def reward_short(self, rewards):
        if not self.train_mode:
            return
        try:
            x = float(rewards)
            rewards = [x] * len(self.episode_history_short)
        except:
            if len(rewards) < len(self.episode_history_short):
                raise Exception(f'Too few rewards {len(rewards)} {len(self.episode_history_short)}')

        for ((state, action), reward) in zip(self.episode_history_short, rewards):
            self.experience.remember_short((state, reward))

        self.episode_history_short = []

    def reward(self, rewards, iteration, threshold_value):

        if not self.train_mode:
            return

        try:
            x = float(rewards)
            rewards = [x] * len(self.episode_history)
        except:
            if len(rewards) < len(self.episode_history):
                raise Exception(f'Too few rewards {len(rewards)} {len(self.episode_history)}')

        self.iteration_counter += 1

        for ((state, action), reward) in zip(self.episode_history, rewards):
            self.experience.remember((state, reward))

        self.episode_history = []

        self.model_fit = True
        loss = []
        if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            loss = self.learn_from_experience(iteration, threshold_value=threshold_value)
        return loss

    def learn_from_experience(self, iteration, threshold_value, drift='false'):
        if self.task_num > 0 and self.grad != 0:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.compute_feature_mat(device)

        experiences = self.experience.get_batch(self.experience_batch_size)
        if (drift == 'true' or not experiences):
            experiences_short = self.experience.get_batch_short(threshold_value=threshold_value, batch_size=1000)
            experiences = experiences_short + experiences


        x, y = zip(*experiences)
        labels = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32).squeeze(dim=-1).to(self.device)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        outputs, _, h_list = self.model(x)

        loss = self.cro_loss(outputs, labels)

        self.opt.zero_grad()

        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param) ** 2

        total_loss = loss + self.lambda_reg * l2_reg
        total_loss.backward()

        if self.net == "lstm":
            if self.task_num > 0 and self.grad != 0:
                kk = 0
                for k, (m, params) in enumerate(self.model.named_parameters()):
                    if k < 6 and len(params.size()) != 1 and 'weight_hh' in m:
                        sz = params.grad.data.size(0)
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                                       self.feature_mat[kk]).view(params.size())  # lstm
                        kk += 1
                    elif k < 6 and len(params.size()) != 1 and 'fc' in m:
                        sz = params.grad.data.size(0)
                        params.grad.data = params.grad.data - torch.mm(self.feature_mat[kk], params.grad.data.view(sz, -1)).view(params.size())

                        kk += 1
                    elif k < 6 and len(params.size()) == 1:
                        params.grad.data.fill_(0)

        self.opt.step()

        if self.grad != 0:
            self.update_SGP(x, self.task_num)
        return [loss, total_loss]

    def compute_feature_mat (self, device):
        self.feature_mat = []
        for i in range(len(self.feature_list)):
            Uf = torch.Tensor(np.dot(self.feature_list[i],np.dot(np.diag(self.importance_list[i]),self.feature_list[i].transpose()))).to(device)
            Uf.requires_grad = False
            self.feature_mat.append(Uf)

    def update_SGP(self, x, task_num):
        _, activations, h_list = self.model(x)
        if self.net == "lstm":
            activation_matrix = activations['lstm'].cpu().detach().numpy()

            concatenated_activations = np.concatenate(activation_matrix, axis=0).reshape(16, len(activation_matrix))
            activation_matrix_fc = activations['fc'].cpu().detach().numpy()
            concatenated_activations_fc = np.concatenate(activation_matrix_fc, axis=0).reshape(1, len(activation_matrix_fc))

            concatenated_activations = [concatenated_activations, concatenated_activations_fc]


        self.feature_list, self.importance_list = self.get_SGP(concatenated_activations, task_num)
        self.task_num += 1

    def get_SGP(self, mat_list, task_id):
        threshold = self.threshold + task_id * self.threshold_inc
        feature_list = self.feature_list
        importance_list = self.importance_list
        if not feature_list:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold[i])+1
                feature_list.append(U[:,0:r])
                importance = ((self.scale_coff+1)*S[0:r])/(self.scale_coff*S[0:r]+ max(S[0:r]))
                importance_list.append(importance)
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                act_proj = np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
                r_old = feature_list[i].shape[1]
                Uc, Sc, Vhc = np.linalg.svd(act_proj, full_matrices=False)
                importance_new_on_old = np.dot(np.dot(feature_list[i].transpose(), Uc[:, 0:r_old]) ** 2,
                                               Sc[0:r_old] ** 2)
                importance_new_on_old = np.sqrt(importance_new_on_old)

                act_hat = activation - act_proj
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    importance = importance_new_on_old
                    importance = ((self.scale_coff+1)*importance)/(self.scale_coff*importance+ max(importance))
                    importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
                    importance_list[i] = importance
                    continue

                importance = np.hstack((importance_new_on_old,S[0:r]))
                importance = ((self.scale_coff+1) * importance)/(self.scale_coff * importance + max(importance))
                importance [0:r_old] = np.clip(importance [0:r_old] + importance_list[i][0:r_old], 0, 1)

                Ui=np.hstack((feature_list[i],U[:,0:r]))
                if Ui.shape[1] > Ui.shape[0] :
                    feature_list[i]=Ui[:,0:Ui.shape[0]]
                    importance_list[i] = importance[0:Ui.shape[0]]
                else:
                    feature_list[i]=Ui
                    importance_list[i] = importance

        return feature_list, importance_list