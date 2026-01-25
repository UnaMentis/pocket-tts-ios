# iOS Simulator Toolchain for cargo builds
# This file is used by cmake-rs when cross-compiling for iOS Simulator
#
# It defines the set_xcode_property function that sentencepiece needs,
# and sets up the iOS Simulator cross-compilation environment.

# Define set_xcode_property macro (from ios-cmake project)
# This is needed by sentencepiece-sys for iOS builds
macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE XCODE_RELVERSION)
    set(XCODE_RELVERSION_I "${XCODE_RELVERSION}")
    if(XCODE_RELVERSION_I STREQUAL "All")
        set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} "${XCODE_VALUE}")
    else()
        set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY}[variant=${XCODE_RELVERSION_I}] "${XCODE_VALUE}")
    endif()
endmacro(set_xcode_property)

# iOS Simulator configuration
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_SYSROOT iphonesimulator)
set(CMAKE_OSX_ARCHITECTURES arm64)

# Disable code signing for library builds
# This is needed because cmake-rs builds static libraries, not apps
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO)
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED NO)
