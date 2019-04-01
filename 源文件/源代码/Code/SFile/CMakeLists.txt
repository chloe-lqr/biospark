cmake_minimum_REQUIRED(VERSION 2.8)

project(sfile)

include(CMakeConfig.txt)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    SET(LINUX true)
    add_definitions(-DLINUX)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(MACOSX true)
    add_definitions(-DMACOSX)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(BUILD_WITH_HDFS)
	add_definitions(-DOPT_HDFS)
	include_directories(src ${HADOOP_ROOT}/include)
        LINK_DIRECTORIES(${HADOOP_ROOT}/lib/native ${JAVA_LIB_DIR})
else(BUILD_WITH_HDFS)
	include_directories(src)
endif(BUILD_WITH_HDFS)

add_subdirectory(src)

add_executable(sfile src/main.cpp)
set_target_properties(sfile PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
set_property(TARGET sfile PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
add_dependencies(sfile c_lib)

if(BUILD_WITH_HDFS)
        target_link_libraries(sfile rt c_lib hdfs jvm)
else(BUILD_WITH_HDFS)
        target_link_libraries(sfile rt c_lib)
endif(BUILD_WITH_HDFS)


install(TARGETS sfile DESTINATION bin)
