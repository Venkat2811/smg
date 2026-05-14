//! Shared request validation for the Responses API.
//!
//! Provides input normalization and validation that is shared across both the
//! HTTP and WebSocket Responses paths.

use openai_protocol::{
    responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest},
    validated::Normalizable,
};
use validator::{Validate, ValidationErrors, ValidationErrorsKind};

pub(crate) fn normalize_and_validate_responses_request(
    request: &mut ResponsesRequest,
) -> Result<(), ValidationErrors> {
    request.normalize();

    match request.validate() {
        Ok(()) => Ok(()),
        Err(errors) if allows_function_call_output_only_continuation(request, &errors) => Ok(()),
        Err(errors) => Err(errors),
    }
}

fn allows_function_call_output_only_continuation(
    request: &ResponsesRequest,
    errors: &ValidationErrors,
) -> bool {
    if request.previous_response_id.is_none() {
        return false;
    }

    let ResponseInput::Items(items) = &request.input else {
        return false;
    };

    if items.is_empty() {
        return false;
    }

    if !items
        .iter()
        .all(|item| matches!(item, ResponseInputOutputItem::FunctionCallOutput { .. }))
    {
        return false;
    }

    let all_errors = errors.errors();
    if all_errors.len() != 1 {
        return false;
    }

    let Some(schema_errors) = all_errors.get("__all__") else {
        return false;
    };

    let ValidationErrorsKind::Field(schema_errors) = schema_errors else {
        return false;
    };

    schema_errors.len() == 1 && schema_errors[0].code == "input_missing_user_message"
}

#[cfg(test)]
mod tests {
    use super::normalize_and_validate_responses_request;
    use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest};

    #[test]
    fn allows_function_call_output_only_continuation_with_previous_response_id() {
        let mut request = ResponsesRequest {
            model: "mock-model".to_string(),
            previous_response_id: Some("resp_prev".to_string()),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "call_123".to_string(),
                output: r#"{"result":345}"#.to_string(),
                status: None,
            }]),
            ..Default::default()
        };

        assert!(normalize_and_validate_responses_request(&mut request).is_ok());
    }

    #[test]
    fn still_rejects_function_call_output_only_without_previous_response_id() {
        let mut request = ResponsesRequest {
            model: "mock-model".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "call_123".to_string(),
                output: r#"{"result":345}"#.to_string(),
                status: None,
            }]),
            ..Default::default()
        };

        let error = normalize_and_validate_responses_request(&mut request).unwrap_err();
        assert!(error
            .to_string()
            .contains("Input items must contain at least one message"));
    }

    #[test]
    fn still_rejects_other_validation_errors_for_continuations() {
        let mut request = ResponsesRequest {
            model: "mock-model".to_string(),
            previous_response_id: Some("resp_prev".to_string()),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "call_123".to_string(),
                output: String::new(),
                status: None,
            }]),
            ..Default::default()
        };

        let error = normalize_and_validate_responses_request(&mut request).unwrap_err();
        assert!(error
            .to_string()
            .contains("Function call output cannot be empty"));
    }
}
