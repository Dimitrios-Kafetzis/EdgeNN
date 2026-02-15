add_executable(test_core test_core.c)
target_link_libraries(test_core PRIVATE edgenn m)
target_include_directories(test_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_core COMMAND test_core)

add_executable(test_dnn_ops test_dnn_ops.c)
target_link_libraries(test_dnn_ops PRIVATE edgenn m)
target_include_directories(test_dnn_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_dnn_ops COMMAND test_dnn_ops)

add_executable(test_rnn_ops test_rnn_ops.c)
target_link_libraries(test_rnn_ops PRIVATE edgenn m)
target_include_directories(test_rnn_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_rnn_ops COMMAND test_rnn_ops)

add_executable(test_transformer_ops test_transformer_ops.c)
target_link_libraries(test_transformer_ops PRIVATE edgenn m)
target_include_directories(test_transformer_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_transformer_ops COMMAND test_transformer_ops)

add_executable(test_runtime test_runtime.c)
target_link_libraries(test_runtime PRIVATE edgenn m)
target_include_directories(test_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_runtime COMMAND test_runtime)
