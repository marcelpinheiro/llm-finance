with source as (
    select * from {{ source('finance', 'financial_statements') }}
),

renamed as (
    select
        id,
        company_id,
        statement_type,
        fiscal_year,
        fiscal_period,
        data,
        created_at
    from source
)

select * from renamed
