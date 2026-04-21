import torch
import torch.nn as nn



class LLM(nn.Module):
    """
    NeuralCDM
    """
    def __init__(self, student_n, exer_n, knowledge_n, Q):
        super(LLM, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim

        # Embedding matrices as Parameters
        self.student_emb = nn.Parameter(torch.zeros(self.emb_num, self.stu_dim))
        self.knowledge_emb = nn.Parameter(
            torch.zeros(self.exer_n, self.knowledge_dim)
        )                     # shape: [exer_n, knowledge_dim]

        self.intercept_emb = nn.Parameter(
            torch.zeros(self.exer_n, 1)
        )                     # shape: [exer_n, 1]

        # Q matrix is usually a fixed mask, keep it as given
        self.Q = Q            # expected shape: [exer_n, knowledge_dim]

        self.apply_clipper()

    def apply_clipper(self):
        with torch.no_grad():
            # Clamps all negative values to zero (non-negative constraint)
            self.knowledge_emb.data.clamp_(min=0) 

    def forward(self, stu_id, exer_id):
        stu_emb = torch.sigmoid(self.student_emb[stu_id])          # [batch, knowledge_dim]
        exer_knowledge = self.knowledge_emb[exer_id]               # [batch, knowledge_dim]
        exer_q = self.Q[exer_id, :]                                # [batch, knowledge_dim]
        intercept = self.intercept_emb[exer_id].squeeze(-1)        # [batch]
        input_x = (stu_emb * exer_knowledge * exer_q).sum(dim=-1) + intercept
        output = torch.sigmoid(input_x)
        return output
    
    def profile(self):
        return torch.sigmoid(self.student_emb)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb[stu_id])
        return stat_emb.data

    def response(self, theta, exer_id):
        """Apply LLM response function to an externally supplied theta (post-sigmoid)."""
        exer_knowledge = self.knowledge_emb[exer_id]
        exer_q = self.Q[exer_id]
        intercept = self.intercept_emb[exer_id].squeeze(-1)
        input_x = (theta * exer_knowledge * exer_q).sum(-1) + intercept
        return torch.sigmoid(input_x)

    def get_exer_params(self):
        return self.knowledge_emb * self.Q



class DINO(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, Q):
        super(DINO, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim


        self.student_emb = nn.Parameter(torch.rand(self.emb_num, self.stu_dim))
        self.params = nn.Parameter(torch.zeros(self.exer_n, 2))
        self.Q = Q

    def apply_clipper(self):
        None

    def forward(self, stu_id, exer_id):
        stu_emb = torch.sigmoid(self.student_emb[stu_id])
        A_masked = stu_emb.masked_fill(~self.Q[exer_id, :].unsqueeze(0), float("-inf"))
        eta, _ = A_masked.max(dim=2)
        eta = eta.squeeze(0)
        params = torch.sigmoid(self.params)[exer_id, ]
        input_x = eta * (params[:, 0] / 2 + 0.5) + (1 - eta) * params[:, 1] / 2
        return input_x

    def response(self, theta, exer_id):
        """Apply DINO response function to an externally supplied theta (post-sigmoid)."""
        A_masked = theta.masked_fill(~self.Q[exer_id], float("-inf"))
        eta = A_masked.max(dim=1).values
        params = torch.sigmoid(self.params)[exer_id]
        return eta * (params[:, 0] / 2 + 0.5) + (1 - eta) * params[:, 1] / 2

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb[stu_id])
        return stat_emb.data

    def get_exer_params(self):
        params = torch.sigmoid(self.params)
        params[:, 0] = params[:, 0] / 2 + 0.5
        params[:, 1] = params[:, 1] / 2
        return params

    def profile(self):
        return torch.sigmoid(self.student_emb)


class DINA(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, Q):
        super(DINA, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim


        self.student_emb = nn.Parameter(torch.rand(self.emb_num, self.stu_dim))
        self.params = nn.Parameter(torch.rand(self.exer_n, 2))
        self.Q = Q

    def apply_clipper(self):
        None

    def forward(self, stu_id, exer_id):
        stu_emb = torch.sigmoid(self.student_emb[stu_id])
        A_masked = stu_emb.masked_fill(~self.Q[exer_id, :].unsqueeze(0), float("inf"))
        eta, _ = A_masked.min(dim=2)
        eta = eta.squeeze(0)
        params = torch.sigmoid(self.params)[exer_id, ]
        input_x = eta * (params[:, 0] / 2 + 0.5) + (1 - eta) * params[:, 1] * 0.5
        return input_x

    def response(self, theta, exer_id):
        """Apply DINA response function to an externally supplied theta (post-sigmoid)."""
        A_masked = theta.masked_fill(~self.Q[exer_id], float("inf"))
        eta = A_masked.min(dim=1).values
        params = torch.sigmoid(self.params)[exer_id]
        return eta * (params[:, 0] / 2 + 0.5) + (1 - eta) * params[:, 1] * 0.5

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb[stu_id])
        return stat_emb.data

    def get_exer_params(self):
        params = torch.sigmoid(self.params)
        params[:, 0] = params[:, 0] / 2 + 0.5
        params[:, 1] = params[:, 1] / 2
        return params

    def profile(self):
        return torch.sigmoid(self.student_emb)

class NeuralCD(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, Q):
        super(NeuralCD, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim

        self.student_emb = nn.Parameter(torch.rand(self.emb_num, self.stu_dim))
        self.k_difficulty = nn.Parameter(torch.rand(self.exer_n, self.knowledge_dim))
        self.e_discrimination = nn.Parameter(torch.rand(self.exer_n, self.knowledge_dim))
        self.Q = Q

        self.fc1 = nn.Linear(self.knowledge_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.5)

        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(fc.weight, 0, 2.0 / fc.weight.shape[1])
        self.fc3.bias.data.fill_(-0.5)

    def apply_clipper(self):
        self.e_discrimination.data.clamp_(0, 1)
        self.fc1.weight.data.clamp_(min=0)
        self.fc2.weight.data.clamp_(min=0)
        self.fc3.weight.data.clamp_(min=0)

    def forward(self, stu_id, exer_id):
        stu_emb = torch.sigmoid(self.student_emb[stu_id])
        k_difficulty = torch.sigmoid(self.k_difficulty[exer_id])
        e_discrimination = torch.sigmoid(self.e_discrimination[exer_id])

        input_x = e_discrimination * (stu_emb - k_difficulty) * self.Q[exer_id]

        input_x = self.drop(torch.sigmoid(self.fc1(input_x)))
        input_x = self.drop(torch.sigmoid(self.fc2(input_x)))
        output = torch.sigmoid(self.fc3(input_x))

        return output.squeeze(-1)

    def response(self, theta, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty[exer_id])
        e_discrimination = torch.sigmoid(self.e_discrimination[exer_id])
        input_x = e_discrimination * (theta - k_difficulty) * self.Q[exer_id]
        input_x = self.drop(torch.sigmoid(self.fc1(input_x)))
        input_x = self.drop(torch.sigmoid(self.fc2(input_x)))
        return torch.sigmoid(self.fc3(input_x)).squeeze(-1)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb[stu_id])
        return stat_emb.data

    def get_exer_params(self):
        k_difficulty = torch.sigmoid(self.k_difficulty)
        e_discrimination = torch.sigmoid(self.e_discrimination)
        return torch.cat((k_difficulty, e_discrimination), dim=1)

    def profile(self):
        return torch.sigmoid(self.student_emb)