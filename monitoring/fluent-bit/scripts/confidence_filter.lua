-- Lua script to filter low confidence predictions
function filter_low_confidence(tag, timestamp, record)
    local confidence = record["confidence"]
    
    if confidence and tonumber(confidence) then
        if tonumber(confidence) < 0.6 then
            -- Add alert flag for low confidence
            record["alert"] = "low_confidence"
            record["alert_level"] = "warning"
            record["alert_message"] = "Low confidence score detected: " .. confidence
            return 2, timestamp, record  -- Forward with modifications
        end
    end
    
    return 1, timestamp, record  -- Forward as-is
end
