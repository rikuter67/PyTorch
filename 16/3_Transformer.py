import torch
sentence = torch.tensor([0, 7, 1, 2, 5, 6, 4, 3])

torch.manual_seed(123)
embed = torch.nn.Embedding(10,16)
embedded_sentence = embed(sentence).detach()

d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)

h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)

x_2 = embedded_sentence[1]
multihead_query_2 = multihead_U_query.matmul(x_2)
multihead_key_2 = multihead_U_key.matmul(x_2)
multihead_value_2 = multihead_U_value.matmul(x_2)
print(multihead_key_2[2])

stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
# print(stacked_inputs.shape)

multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
print(multihead_keys.shape)

multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_keys.shape)

print(multihead_keys[2, 1])

multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)

multihead_z_2 = torch.rand(8, 16)

linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
print(context_vector_2.shape)