import unittest

case=unittest.TestCase()
case.assertEqual(1,1)

list=[1,2,3,1,2,5]
set_new=set()
print(len(list))
for i in range(len(list)):
    set_new.add(list[i])
    # print(list[0])
#     if list[i] not in list_new:
#         list_new.append(list[i])
# print(list_new)
print(set_new)



