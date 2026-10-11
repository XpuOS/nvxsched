set_property(TARGET jsoncpp_static PROPERTY POSITION_INDEPENDENT_CODE ON)
add_hal_lib(cuda LINK_LIBRARIES jsoncpp_static cuxtra)
add_shim_lib(cuda
    SOFTLINKS_LINUX libcuda.so.1 libcuda.so
    TARGET_NAME_WINDOWS "nvcuda"
)
add_platform_test()

add_subdirectory(profile)
