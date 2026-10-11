
file(GLOB_RECURSE CUDA_PROFILE_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(profilecuda SHARED ${CUDA_PROFILE_SRCS})
add_library(XSched::profilecuda ALIAS profilecuda)
target_include_directories(profilecuda PUBLIC
    $<BUILD_INTERFACE:${XSCHED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:preempt,INTERFACE_INCLUDE_DIRECTORIES>>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:halcuda,INTERFACE_INCLUDE_DIRECTORIES>>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(profilecuda PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)
target_link_libraries(profilecuda
    PRIVATE utils
    PRIVATE protocol
)
target_link_options(profilecuda PRIVATE
    -Wl,--exclude-libs,ALL
)
set_target_properties(profilecuda PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib/profile
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS profilecuda EXPORT XSchedTargets LIBRARY DESTINATION lib/profile)

foreach(softlink libcuda.so.1 libcuda.so)
    install(CODE "
        execute_process(
            COMMAND ln -s libprofilecuda.so ${softlink}
            WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib/profile
        )
        message(\"-- Installing: shim softlink created: ${softlink}\")
    ")
endforeach()
