import torch
# 0: can, 7: you, 1: help, 2: me, 5 to, 6: translate, 4: this, 3: sentence
sentence = torch.tensor([0, 7, 1, 2, 5, 6, 4, 3])
# print(sentence)
# : tensor([0, 7, 1, 2, 5, 6, 4, 3])

torch.manual_seed(123)
embed = torch.nn.Embedding(10,16)
embedded_sentence = embed(sentence).detach()
# print(embedded_sentence.shape)
# print(embedded_sentence)

omega = torch.empty(8, 8)
# for文は効率的ではない
for i, x_i in enumerate(embedded_sentence):
    for j, x_j in enumerate(embedded_sentence):
        omega[i, j] = torch.dot(x_i, x_j)

omega_mat = embedded_sentence.matmul(embedded_sentence.T)
# print(torch.allclose(omega_mat, omega))

import torch.nn.functional as F
attention_weights = F.softmax(omega, dim=1)
# print(attention_weights.shape)
# print(attention_weights.sum(dim=1))

x_2 = embedded_sentence[1, :]
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 += attention_weights[1, j] * x_j
# print(context_vec_2)

context_vectors = torch.matmul(attention_weights, embedded_sentence)
# print(torch.allclose(context_vec_2, context_vectors[1]))

torch.manual_seed(123)
d = embedded_sentence.shape[1]
U_query = torch.rand(d, d)
U_key = torch.rand(d, d)
U_value = torch.rand(d, d)

x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)

keys = U_key.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T

# print(torch.allclose(key_2, keys[1]))
# print(torch.allclose(value_2, values[1]))

omega_23 = query_2.dot(keys[2])
# print(omega_23)
omega_2 = query_2.matmul(keys.T)
# print(omega_2)

attention_weights_2 = F.softmax(omega_2 / d**0.5, dim=0)
# print(attention_weights_2)

context_vector_2 = attention_weights_2.matmul(values)
print(context_vector_2)