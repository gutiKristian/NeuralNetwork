project(PV021)
set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} "src/main.cpp" "src/NeuralNet.h" "src/NeuralNet.cpp" "src/Layer.h"  "src/Activations.h"  "src/Activations.cpp")