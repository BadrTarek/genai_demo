# Customer Onboarding Requirements

## Overview

The system must allow internal users to create and manage customer profiles.

## Functional Requirements

- An employee must provide the customer's legal name, email address, phone number, and country.
- The system must validate the customer email address.
- The system must prevent duplicate customers with the same email address.
- New customers must initially have the status Pending.
- A customer must be approved before they can create orders.
- The system must record who approved the customer and when.
- Employees must be able to search customers by name, email, or status.

## Audit Requirements

- Customer creation must be recorded in the audit log.
- Customer updates must record the previous and new values.
- Audit records must not be deleted by regular employees.