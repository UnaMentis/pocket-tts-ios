# iOS build initialization script
# Defines helper functions needed by sentencepiece-sys
#
# This script is included via CMAKE_PROJECT_INCLUDE_BEFORE to provide
# the set_xcode_property function that sentencepiece uses for iOS builds.
# The function is typically provided by ios.toolchain.cmake, but cmake-rs
# doesn't use that toolchain file.

# Define set_xcode_property macro if it doesn't exist
# This is from ios-cmake project: https://github.com/leetal/ios-cmake
if(NOT COMMAND set_xcode_property)
    macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE XCODE_RELVERSION)
        set(XCODE_RELVERSION_I "${XCODE_RELVERSION}")
        if(XCODE_RELVERSION_I STREQUAL "All")
            set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} "${XCODE_VALUE}")
        else()
            set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY}[variant=${XCODE_RELVERSION_I}] "${XCODE_VALUE}")
        endif()
    endmacro(set_xcode_property)
endif()
