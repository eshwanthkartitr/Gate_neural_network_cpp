include(FetchContent)
FetchContent_Declare(
  OpenCL-SDK
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
  GIT_TAG        v2024.04.17
)
FetchContent_MakeAvailable(OpenCL-SDK)
target_link_libraries(MyApp PRIVATE OpenCL::OpenCL)
