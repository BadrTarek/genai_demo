# User Authentication Requirements

## Overview

The system must provide secure authentication for registered users.

## Functional Requirements

- Users must sign in using an email address and password.
- Email addresses must be unique and case-insensitive.
- Passwords must contain at least 12 characters.
- Passwords must include uppercase, lowercase, number, and special character.
- Users must receive an error message when authentication fails.
- The system must lock an account after five failed attempts within 15 minutes.
- Locked accounts must automatically unlock after 30 minutes.
- Users must be able to reset their password using a verified email address.

## Security Requirements

- Passwords must never be stored as plain text.
- Authentication requests must use HTTPS.
- User sessions must expire after 30 minutes of inactivity.