import random
import torch
import torch.nn as nn
from torch import optim

# 랜덤 시드 설정
torch.manual_seed(0)
# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 데이터 (영어-한국어 문장 쌍)
raw = ["I feel hungry.	나는 배가 고프다.",
       "Pytorch is very easy.	파이토치는 매우 쉽다.",
       "Pytorch is a framework for deep learning.	파이토치는 딥러닝을 위한 프레임워크이다.",
       "Pytorch is very clear to use.	파이토치는 사용하기 매우 직관적이다."]

# 문장 시작(SOS)과 끝(EOS)을 나타내는 토큰
SOS_token = 0
EOS_token = 1

# 어휘 정보 관리를 위한 클래스 정의
class Vocab:
    def __init__(self):
        self.vocab2index = {"<SOS>": SOS_token, "<EOS>": EOS_token}  # 단어 → 인덱스
        self.index2vocab = {SOS_token: "<SOS>", EOS_token: "<EOS>"}  # 인덱스 → 단어
        self.vocab_count = {}  # 단어 빈도수
        self.n_vocab = len(self.vocab2index)  # 어휘 크기

    # 새로운 단어를 어휘집에 추가
    def add_vocab(self, sentence):
        for word in sentence.split(" "):  # 문장을 단어로 분리
            if word not in self.vocab2index:
                self.vocab2index[word] = self.n_vocab
                self.index2vocab[self.n_vocab] = word
                self.vocab_count[word] = 1
                self.n_vocab += 1
            else:
                self.vocab_count[word] += 1

# 긴 문장을 필터링하는 함수
def filter_pair(pair, source_max_length, target_max_length):
    return len(pair[0].split(" ")) < source_max_length and len(pair[1].split(" ")) < target_max_length

# 데이터 전처리 함수
def preprocess(corpus, source_max_length, target_max_length):
    print("Reading corpus...")
    pairs = []
    for line in corpus:
        pairs.append([s for s in line.strip().lower().split("\t")])  # 소스와 타겟 문장 분리
    print("Read {} sentence pairs".format(len(pairs)))

    # 문장 길이 제한
    pairs = [pair for pair in pairs if filter_pair(pair, source_max_length, target_max_length)]
    print("Trimmed to {} sentence pairs".format(len(pairs)))

    # 어휘집 생성
    source_vocab = Vocab()
    target_vocab = Vocab()

    print("Counting words...")
    for pair in pairs:
        source_vocab.add_vocab(pair[0])  # 소스 어휘 추가
        target_vocab.add_vocab(pair[1])  # 타겟 어휘 추가
    print("Source vocab size =", source_vocab.n_vocab)
    print("Target vocab size =", target_vocab.n_vocab)

    return pairs, source_vocab, target_vocab

# 간단한 인코더 정의
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # 단어 임베딩
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU 레이어

    def forward(self, x, hidden):
        x = self.embedding(x).view(1, 1, -1)  # 임베딩
        x, hidden = self.gru(x, hidden)  # GRU 통과
        return x, hidden

# 간단한 디코더 정의
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)  # 단어 임베딩
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU 레이어
        self.out = nn.Linear(hidden_size, output_size)  # 출력 레이어
        self.softmax = nn.LogSoftmax(dim=1)  # 소프트맥스 함수

    def forward(self, x, hidden):
        x = self.embedding(x).view(1, 1, -1)  # 임베딩
        x, hidden = self.gru(x, hidden)  # GRU 통과
        x = self.softmax(self.out(x[0]))  # 출력 생성
        return x, hidden

# 문장을 인덱스 텐서로 변환
def tensorize(vocab, sentence):
    indexes = [vocab.vocab2index[word] for word in sentence.split(" ")]
    indexes.append(vocab.vocab2index["<EOS>"])  # EOS 추가
    return torch.Tensor(indexes).long().to(device).view(-1, 1)

# 학습 함수
def train(pairs, source_vocab, target_vocab, encoder, decoder, n_iter, print_every=1000, learning_rate=0.01):
    loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_batch = [random.choice(pairs) for _ in range(n_iter)]
    training_source = [tensorize(source_vocab, pair[0]) for pair in training_batch]
    training_target = [tensorize(target_vocab, pair[1]) for pair in training_batch]

    criterion = nn.NLLLoss()  # 손실 함수

    for i in range(1, n_iter + 1):
        source_tensor = training_source[i - 1]
        target_tensor = training_target[i - 1]

        encoder_hidden = torch.zeros([1, 1, encoder.hidden_size]).to(device)  # 초기 히든 상태

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        source_length = source_tensor.size(0)
        target_length = target_tensor.size(0)

        loss = 0

        # 인코더 학습
        for enc_input in range(source_length):
            _, encoder_hidden = encoder(source_tensor[enc_input], encoder_hidden)

        decoder_input = torch.Tensor([[SOS_token]]).long().to(device)
        decoder_hidden = encoder_hidden  # 인코더 출력 → 디코더 입력

        # 디코더 학습
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])  # 손실 계산
            decoder_input = target_tensor[di]  # Teacher Forcing

        loss.backward()  # 역전파

        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_iter = loss.item() / target_length
        loss_total += loss_iter

        if i % print_every == 0:
            loss_avg = loss_total / print_every
            loss_total = 0
            print("[{} - {}%] loss = {:05.4f}".format(i, i / n_iter * 100, loss_avg))

# 모델 평가 함수
def evaluate(pairs, source_vocab, target_vocab, encoder, decoder, target_max_length):
    for pair in pairs:
        print(">", pair[0])  # 입력 문장
        print("=", pair[1])  # 실제 정답
        source_tensor = tensorize(source_vocab, pair[0])
        source_length = source_tensor.size()[0]
        encoder_hidden = torch.zeros([1, 1, encoder.hidden_size]).to(device)

        for ei in range(source_length):
            _, encoder_hidden = encoder(source_tensor[ei], encoder_hidden)

        decoder_input = torch.Tensor([[SOS_token]]).long().to(device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(target_max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, top_index = decoder_output.data.topk(1)  # 가장 높은 확률의 단어 선택
            if top_index.item() == EOS_token:  # EOS 검사
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(target_vocab.index2vocab[top_index.item()])

            decoder_input = top_index.squeeze().detach()

        predict_words = decoded_words
        predict_sentence = " ".join(predict_words)
        print("<", predict_sentence)  # 예측된 문장
        print("")

# 문장 최대 길이 정의
SOURCE_MAX_LENGTH = 10
TARGET_MAX_LENGTH = 12

# 데이터 전처리
load_pairs, load_source_vocab, load_target_vocab = preprocess(raw, SOURCE_MAX_LENGTH, TARGET_MAX_LENGTH)
print(random.choice(load_pairs))
# Reading corpus...
# Read 4 sentence pairs
# Trimmed to 4 sentence pairs
# Counting words...
# Source vocab size = 17
# Target vocab size = 13
# ['pytorch is very easy.', '파이토치는 매우 쉽다.']
# [1000 - 20.0%] loss = 0.7380
# [2000 - 40.0%] loss = 0.1098
# [3000 - 60.0%] loss = 0.0356
# [4000 - 80.0%] loss = 0.0187
# [5000 - 100.0%] loss = 0.0126

# 인코더와 디코더 선언
enc_hidden_size = 16
dec_hidden_size = enc_hidden_size
enc = Encoder(load_source_vocab.n_vocab, enc_hidden_size).to(device)
dec = Decoder(dec_hidden_size, load_target_vocab.n_vocab).to(device)

# 학습 실행
train(load_pairs, load_source_vocab, load_target_vocab, enc, dec, 5000, print_every=1000)

# 평가 실행
evaluate(load_pairs, load_source_vocab, load_target_vocab, enc, dec, TARGET_MAX_LENGTH)
# [1000 - 20.0%] loss = 0.7380
# [2000 - 40.0%] loss = 0.1098
# [3000 - 60.0%] loss = 0.0356
# [4000 - 80.0%] loss = 0.0187
# [5000 - 100.0%] loss = 0.0126
# > i feel hungry.
# = 나는 배가 고프다.
# < 나는 배가 고프다. <EOS>

# > pytorch is very easy.
# = 파이토치는 매우 쉽다.
# < 파이토치는 매우 쉽다. <EOS>

# > pytorch is a framework for deep learning.
# = 파이토치는 딥러닝을 위한 프레임워크이다.
# < 파이토치는 딥러닝을 위한 프레임워크이다. <EOS>

# > pytorch is very clear to use.
# = 파이토치는 사용하기 매우 직관적이다.
# < 파이토치는 사용하기 매우 직관적이다. <EOS>