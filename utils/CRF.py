import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
torch.manual_seed(47)


# CRF Model
class CRF(nn.Module):
    """
    条件随机场的实现。
    使用前向-后向算法计算输入的对数似然。参考论文 Neural Architectures for Named Entity Recognition 。
    基于Python3.10.9和torch-1.13.1

    """

    def __init__(
            self,
            num_tags: int,
            batch_first: bool = False,
    ) -> None:
        """初始化CRF的参数

        Args:
            num_tags (int): 标签数量
            batch_first (bool, optional): 是否batch维度在前，默认为False
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # 转移分数(A)，表示两个标签之间转移的得分
        # transitions[i,j] 表示由第i个标签转移到第j个标签的得分(可以理解为可能性/置信度)
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        # 新引入了两个状态：start和end
        # 从start状态开始转移的分数
        self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
        # 转移到end状态的分数
        self.end_transitions = nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def __repr__(self):
        return f"CRF(num_tags={self.num_tags})"

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: torch.ByteTensor | None = None,
            reduction: str = 'sum'
    ) -> torch.Tensor:
        """计算给定的标签序列tags的负对数似然

        Args:
            emissions (torch.Tensor):  发射分数P 形状 (seq_len, batch_size, num_tags), 代表序列中每个单词产生每个标签的得分
            tags (torch.LongTensor): 标签序列 如果batch_first=False 形状 (seq_len, batch_size) ，否则 形状为 (batch_size, seq_len)
            mask (torch.ByteTensor | None, optional): 表明哪些元素为填充符，和tags的形状一致。  如果batch_first=False  形状 (seq_len, batch_size) ，否则 形状为 (batch_size, seq_len)
                默认为None，表示没有填充符。
            reduction (str): 汇聚函数： none|sum|mean|token_mean 。 none：不应用汇聚函数；

        Returns:
            torch.Tensor: 输入tags的负对数似然
        """

        if mask is None:
            # mask 取值为0或1，这里1表示有效标签，默认都为有效标签
            mask = torch.ones_like(tags)

        if self.batch_first:
            # 转换为seq维度在前的形式
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # 计算标签序列tags的得分
        score = self._compute_score(emissions, tags, mask)
        # 计算划分函数 partition Z(x)
        partition = self._compute_partition(emissions, mask)
        # negative log likelihood
        nllh = partition - score

        if reduction == 'none':
            return nllh
        if reduction == 'sum':
            return nllh.sum()
        if reduction == 'mean':
            return nllh.mean()
        # 否则为 'token_mean'
        return nllh.sum() / mask.type_as(emissions).sum()

    def _compute_score(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: torch.ByteTensor,
    ) -> torch.Tensor:
        """计算标签序列tags的得分


        Args:
            emissions (torch.Tensor): 发射分数P 形状 (seq_len, batch_size, num_tags)
            tags (torch.LongTensor): 标签序列 形状 (seq_len, batch_size)
            mask (torch.ByteTensor): 表明哪些元素为填充符 形状 (seq_len, batch_size)

        Returns:
            torch.Tensor: 批次内标签tags的得分， 形状(batch_size,)
        """

        seq_len, batch_size = tags.shape
        # first_tags (batch_size,)
        first_tags = tags[0]

        # 由start标签转移到批次内所有标签序列第一个标签的得分
        score = self.start_transitions[first_tags]
        # 加上 批次内第一个(index=0)发射得分，即批次内第0个输入产生批次内对应第0个标签的得分

        score += emissions[0, torch.arange(batch_size), first_tags]

        mask = mask.type_as(emissions)  # 类型保持一致
        # 这里的index从1开始，也就是第二个位置开始
        for i in range(1, seq_len):
            # 第i-1个标签转移到第i个标签的得分 + 第i个单词产生第i个标签的得分
            # 乘以mask[i]不需要计算填充单词的得分
            # score 形状(batch_size,)
            score += (
                             self.transitions[tags[i - 1], tags[i]]
                             + emissions[i, torch.arange(batch_size), tags[i]]
                     ) * mask[i]

        # last_tags = tags[-1] × 这是错误的！，因为可能包含填充单词
        valid_last_idx = mask.long().sum(dim=0) - 1  # 有效的最后一个索引
        last_tags = tags[valid_last_idx, torch.arange(batch_size)]

        # 最后加上最后一个标签转移到end标签的转移得分
        score += self.end_transitions[last_tags]
        return score

    def _compute_partition(
            self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        """利用CRF的前向算法计算partition的分数

        Args:
            emissions (torch.Tensor): 发射分数P 形状 (seq_len, batch_size, num_tags)
            mask (torch.ByteTensor): 表明哪些元素为填充符  (seq_len, batch_size)

        Returns:
            torch.Tensor: 批次内的partition分数 形状(batch_size,)
        """

        seq_len = emissions.shape[0]
        # score (batch_size, num_tags) 对于每个批次来说，第i个元素保存到目前为止以i结尾的所有可能序列的得分
        score = self.start_transitions.unsqueeze(0) + emissions[0]

        for i in range(1, seq_len):
            # broadcast_score: (batch_size, num_tags, 1) = (batch_size, pre_tag, current_tag)
            # 所有可能的当前标签current_tag广播
            broadcast_score = score.unsqueeze(2)
            # 广播成 (batch_size, 1, num_tags)
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            # (batch_size, num_tags, num_tags) = (batch_size, num_tags, 1) + (num_tags, num_tags) + (batch_size, 1, num_tags)
            current_score = broadcast_score + self.transitions + broadcast_emissions
            # 在前一时间步标签上求和  -> (batch_size, num_tags)
            # 对于每个批次来说，第i个元素保存到目前为止以i结尾的所有可能标签序列的得分
            current_score = torch.logsumexp(current_score, dim=1)
            # mask[i].unsqueeze(1) -> (batch_size, 1)
            # 只有mask[i]是有效标签的current_score才将值设置到score中，否则保持原来的score
            score = torch.where(mask[i].bool().unsqueeze(1), current_score, score)

        # 加上到end标签的转移得分 end_transitions本身保存的是所有的标签到end标签的得分
        # score (batch_size, num_tags)
        score += self.end_transitions
        # 在所有的标签上求(logsumexp)和
        # return (batch_size,)
        return torch.logsumexp(score, dim=1)

    def decode(
            self, emissions: torch.Tensor, mask: torch.ByteTensor = None
    ) -> list[list[int]]:
        """使用维特比算法找到最有可能的序列

        Args:
            emissions (torch.Tensor):  发射分数P 形状 (seq_len, batch_size, num_tags), 代表序列中每个单词产生每个标签的得分
            mask (torch.ByteTensor | None, optional): 表明哪些元素为填充符。  如果batch_first=False  形状 (seq_len, batch_size) ，否则 形状为 (batch_size, seq_len)
                默认为None，表示没有填充符。

        Returns:
            list[list[int]]: 批次内的最佳标签序列
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)

        if self.batch_first:
            # 转换为seq维度在前的形式
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi(emissions, mask)

    def _viterbi(
            self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> list[list[int]]:
        """维特比算法的实现

        Args:
            emissions (torch.Tensor): 发射分数P 形状 (seq_len, batch_size, num_tags)
            mask (torch.ByteTensor): 表明哪些元素为填充符 形状 (seq_len, batch_size)

        Returns:
            list[list[int]]: 批次内的最佳标签序列
        """
        seq_len, batch_size = mask.shape
        # 由start到当前时间步所有标签的转移得分 + 批次内所有当前时间步产生所有标签的发射得分
        # (num_tags,) +  (batch_size, num_tags) -> (batch_size, num_tags)
        # score 形状 (batch_size, num_tags) 保存了当前位置，到每个tag的最佳累计得分： 前一累计得分+转移得分+发射得分
        score = self.start_transitions + emissions[0]
        # 保存了目前为止到当前时间步所有标签的最佳候选路径 最终有seq_len-1个(batch_size, num_tags)的Tensor
        history: list[torch.Tensor] = []

        for i in range(1, seq_len):
            # broadcast_score: (batch_size, num_tags, 1) = (batch_size, pre_tag, current_tag)
            # 所有可能的当前标签current_tag广播
            broadcast_score = score.unsqueeze(2)
            # 广播成 (batch_size, 1, num_tags)
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            # (batch_size, num_tags, num_tags) = (batch_size, num_tags, 1) + (num_tags, num_tags) + (batch_size, 1, num_tags)
            current_score = broadcast_score + self.transitions + broadcast_emissions
            # 计算前一时间步到当前时间步的某个标签的最佳得分
            # best_score 形状 (batch_size, num_tags) indices 形状 (batch_size, num_tags)
            # indices 相当于 torch.argmax(current_score, dim=1) ，得到产生最大值对应的索引
            best_score, indices = torch.max(current_score, dim=1)

            # mask[i].unsqueeze(1) -> (batch_size, 1)
            # 只有mask[i]是有效标签的best_score才将值设置到score中，否则保持原来的score
            score = torch.where(mask[i].bool().unsqueeze(1), best_score, score)
            # 记录得到最佳得分的前一个索引
            history.append(indices)

        # 加上到end标签的转移得分 end_transitions本身保存的是所有的标签到end标签的得分
        # score (batch_size, num_tags)
        score += self.end_transitions
        # 计算出最后一个时间步到end标签的最大得分 以及对应的索引(tag)
        # best_score 形状(batch_size,)  indices 形状(batch_size,)
        best_score, indices = torch.max(score, dim=1)
        # 序列最后有效标签的个数
        seq_end_tags = mask.long().sum(dim=0) - 1
        # 保存需要返回的结果
        best_paths: list[list[int]] = []

        # 因为批次内每个样本的最后一个有效标签可能不同，因此需要写成for循环
        for i in range(batch_size):
            best_last_tag = indices[i]
            # 通过item()变成普通int
            this_path = [best_last_tag.item()]
            # history 有 seq_len-1个(batch_size, num_tags)， 但是是顺序添加的，history[: seq_end_tags[i]]取有效的history，再逆序
            for hist in reversed(history[: seq_end_tags[i]]):
                # 先取批次内第i个样本的路径，再取到this_path[-1]，即best_last_tag的最佳标签
                best_last_tag = hist[i][this_path[-1]]
                # 转换为int加入到最佳路径中
                this_path.append(best_last_tag.item())
            # 这里是通过回溯的方法添加的最佳路径，因此最后还需要reversed逆序，变回顺序的，存入best_tags列表
            this_path.reverse()
            best_paths.append(this_path)

        return best_paths


