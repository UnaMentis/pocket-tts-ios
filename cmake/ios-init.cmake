# iOS build initialization script
# Defines helper functions needed by sentencepiece-sys

# Define set_xcode_property if it doesn't exist
if(NOT COMMAND set_xcode_property)
    function(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE)
        set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} ${XCODE_VALUE})
    endfunction()
endif()
