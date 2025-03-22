# Experiment 1

def count_occurrences(lst, element):
     count = 0
     for item in lst:
         if item == element:
             count += 1
     return count
 
a=count_occurrences([1,4,6,8,4,6,4,8,4,1],1)
print(a)
# 

#Experiment 2

# def bubble_sort(arr):
#     n = len(arr)
#     for i in range(n - 1):
#         for j in range(0, n - i - 1):
#             if arr[j] > arr[j + 1]:
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
# # Test the bubble sort function
# arr = [64, 34, 25, 12, 22, 11, 90]
# bubble_sort(arr)
# print("Sorted array is:", arr)


#Experiment 3

# def insertion_sort(arr):
#     for i in range(1, len(arr)):
#         key = arr[i]
#         j = i - 1
#         while j >= 0 and key < arr[j]:
#             arr[j + 1] = arr[j]
#             j -= 1
#         arr[j + 1] = key
#  
# # Test the insertion sort function
# arr = [12, 11, 13, 5, 6]
# insertion_sort(arr)
# print("Sorted array is:", arr)
