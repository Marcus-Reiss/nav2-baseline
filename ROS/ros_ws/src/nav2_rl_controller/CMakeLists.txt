cmake_minimum_required(VERSION 3.8)
project(nav2_rl_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========================
# Dependências
# ========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ========================
# Geração da interface (srv)
# ========================
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/RLInfer.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

# ========================
# Biblioteca principal (plugin)
# ========================
set(LIB_NAME nav2_rl_controller_plugin)

add_library(${LIB_NAME} SHARED
  src/rl_controller.cpp
)

# make include dir available to the target during build & install
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${LIB_NAME}
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  pluginlib
  geometry_msgs
  nav_msgs
  tf2_ros
  rosidl_default_runtime
)

# Link generated messages/typesupport
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(${LIB_NAME} ${cpp_typesupport_target})

# ========================
# Exportar plugin e instalar
# ========================
pluginlib_export_plugin_description_file(nav2_core plugins.xml)

install(
  TARGETS ${LIB_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# install headers
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
